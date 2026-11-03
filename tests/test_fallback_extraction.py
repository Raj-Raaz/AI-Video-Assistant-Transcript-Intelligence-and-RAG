import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from core.llm_utils import get_llm, RateLimitError, GROQ_ENDPOINT, MISTRAL_ENDPOINT
from core.extractor import extract_information


def response(url, status=200, content='answer'):
    return httpx.Response(status, request=httpx.Request('POST', url),
                         json={'choices': [{'message': {'content': content}}]})


@patch.dict(os.environ, {'GROQ_API_KEY': 'groq-test', 'MISTRAL_API_KEY': 'mistral-test'}, clear=True)
class FallbackTests(unittest.TestCase):
    def chain(self):
        return ChatPromptTemplate.from_messages([('human', '{text}')]) | get_llm(json_mode=True)

    @patch('core.llm_utils.time.sleep')
    @patch('core.llm_utils.httpx.post')
    def test_fallback_after_four_attempts(self, post, sleep):
        post.side_effect = [response(GROQ_ENDPOINT, 429) for _ in range(4)] + [response(MISTRAL_ENDPOINT)]
        self.assertEqual(self.chain().invoke({'text': 'transcript'}), 'answer')
        self.assertEqual(post.call_count, 5)
        first = post.call_args_list[0]
        last = post.call_args_list[-1]
        self.assertEqual(last.args[0], MISTRAL_ENDPOINT)
        self.assertEqual(last.kwargs['json']['model'], 'mistral-small-latest')
        self.assertEqual(last.kwargs['headers']['Authorization'], 'Bearer mistral-test')
        self.assertEqual(last.kwargs['json']['messages'], first.kwargs['json']['messages'])
        self.assertEqual(last.kwargs['json']['response_format'], {'type': 'json_object'})
        self.assertIn('max_tokens', last.kwargs['json'])
        self.assertNotIn('max_completion_tokens', last.kwargs['json'])

    @patch('core.llm_utils.time.sleep')
    @patch('core.llm_utils.httpx.post')
    def test_both_providers_limited(self, post, sleep):
        post.side_effect = [response(GROQ_ENDPOINT, 429) for _ in range(4)] + [response(MISTRAL_ENDPOINT, 429) for _ in range(4)]
        with self.assertRaisesRegex(RateLimitError, 'Mistral'):
            self.chain().invoke({'text': 'transcript'})
        self.assertEqual(post.call_count, 8)

    @patch('core.llm_utils.time.sleep')
    @patch('core.llm_utils.httpx.post')
    def test_missing_fallback_key(self, post, sleep):
        post.return_value = response(GROQ_ENDPOINT, 429)
        with patch.dict(os.environ, {'MISTRAL_API_KEY': ''}):
            with self.assertRaisesRegex(RateLimitError, 'not configured'):
                self.chain().invoke({'text': 'transcript'})
        self.assertEqual(post.call_count, 4)

    @patch('core.llm_utils.httpx.post')
    def test_authentication_errors_do_not_trigger_fallback(self, post):
        post.return_value = response(GROQ_ENDPOINT, 401)
        with self.assertRaises(httpx.HTTPStatusError):
            self.chain().invoke({'text': 'transcript'})
        self.assertEqual(post.call_count, 1)


class ExtractionTests(unittest.TestCase):
    def test_all_fields_in_one_request(self):
        expected = {'action_items': 'No action items found.',
                    'key_decisions': 'No key decisions found.',
                    'questions': 'No open questions found.'}
        calls = []
        def complete(prompt):
            calls.append(prompt)
            return json.dumps(expected)
        with patch('core.extractor.get_llm', return_value=RunnableLambda(complete)):
            self.assertEqual(extract_information('A news video.'), expected)
        self.assertEqual(len(calls), 1)

    def test_invalid_or_missing_fields_are_not_silently_accepted(self):
        for output in ['not JSON', '{}', '{"action_items": []}']:
            with self.subTest(output=output):
                with patch('core.extractor.get_llm', return_value=RunnableLambda(lambda _: output)):
                    with self.assertRaises(ValueError):
                        extract_information('transcript')
