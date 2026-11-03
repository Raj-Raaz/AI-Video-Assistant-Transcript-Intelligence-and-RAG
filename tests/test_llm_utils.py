import unittest
from unittest.mock import Mock, patch
import httpx
from langchain_core.runnables import RunnableLambda
from core.llm_utils import invoke_with_retry, RateLimitError
from core import summarize


def error(status, headers=None):
    response = httpx.Response(status, headers=headers, request=httpx.Request('POST', 'https://example.com'))
    return httpx.HTTPStatusError('test', request=response.request, response=response)


class RetryTests(unittest.TestCase):
    @patch('core.llm_utils.time.sleep')
    def test_recovery_respects_retry_after(self, sleep):
        chain = Mock()
        chain.invoke.side_effect = [error(429, {'Retry-After': '2'}), 'ok']
        self.assertEqual(invoke_with_retry(chain, 'input'), 'ok')
        sleep.assert_called_once_with(2)

    @patch('core.llm_utils.time.sleep')
    def test_exhaustion_is_bounded(self, sleep):
        chain = Mock()
        chain.invoke.side_effect = error(429)
        with self.assertRaises(RateLimitError):
            invoke_with_retry(chain, 'input')
        self.assertEqual(chain.invoke.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    @patch('core.llm_utils.time.sleep')
    def test_long_server_delay_stops_without_early_retry(self, sleep):
        chain = Mock()
        chain.invoke.side_effect = error(429, {'Retry-After': '120'})
        with self.assertRaises(RateLimitError):
            invoke_with_retry(chain, 'input')
        sleep.assert_not_called()
        self.assertEqual(chain.invoke.call_count, 1)

    def test_other_http_errors_propagate(self):
        chain = Mock()
        chain.invoke.side_effect = error(401)
        with self.assertRaises(httpx.HTTPStatusError):
            invoke_with_retry(chain, 'input')
        self.assertEqual(chain.invoke.call_count, 1)

    def test_prompts_and_single_chunk_request_count(self):
        calls = []
        def respond(prompt):
            calls.append(prompt)
            return 'result'
        with patch.object(summarize, 'get_llm', return_value=RunnableLambda(respond)):
            self.assertEqual(summarize.generate_title('some transcript'), 'result')
            self.assertEqual(summarize.summarize_text('some transcript'), 'result')
        self.assertEqual(len(calls), 2)

    def test_multi_chunk_summary(self):
        calls = []
        def respond(prompt):
            calls.append(prompt)
            return 'result'
        with patch.object(summarize, 'get_llm', return_value=RunnableLambda(respond)):
            with patch.object(summarize, 'split_text', return_value=['one', 'two']):
                self.assertEqual(summarize.summarize_text('input'), 'result')
        self.assertEqual(len(calls), 3)

if __name__ == '__main__':
    unittest.main()
