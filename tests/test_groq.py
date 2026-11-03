import os
import unittest
from unittest.mock import patch
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_utils import get_llm, invoke_with_retry, GROQ_ENDPOINT


class GroqTests(unittest.TestCase):
    @patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'})
    @patch('core.llm_utils.httpx.post')
    def test_chain_sends_groq_model_and_messages(self, post):
        post.return_value = httpx.Response(200, request=httpx.Request('POST', GROQ_ENDPOINT),
            json={'choices': [{'message': {'content': 'A title'}}]})
        chain = ChatPromptTemplate.from_messages([('system', 'Create title'), ('human', '{text}')]) | get_llm() | StrOutputParser()
        self.assertEqual(invoke_with_retry(chain, {'text': 'Transcript'}), 'A title')
        args, kwargs = post.call_args
        self.assertEqual(args[0], GROQ_ENDPOINT)
        self.assertEqual(kwargs['json']['model'], 'openai/gpt-oss-120b')
        self.assertEqual(kwargs['json']['messages'][1], {'role': 'user', 'content': 'Transcript'})
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer test-key')

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_key(self):
        with self.assertRaisesRegex(RuntimeError, 'GROQ_API_KEY'):
            get_llm()

    @patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'})
    @patch('core.llm_utils.time.sleep')
    @patch('core.llm_utils.httpx.post')
    def test_provider_429_retries(self, post, sleep):
        request = httpx.Request('POST', GROQ_ENDPOINT)
        post.side_effect = [httpx.Response(429, request=request, headers={'Retry-After': '1'}),
            httpx.Response(200, request=request, json={'choices': [{'message': {'content': 'ok'}}]})]
        chain = ChatPromptTemplate.from_messages([('human', '{text}')]) | get_llm()
        self.assertEqual(invoke_with_retry(chain, {'text': 'Hello'}), 'ok')
        sleep.assert_called_once_with(1)
