import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.result = dict(title='Test video', summary='A summary', transcript='Full transcript',
                           action_items='No action items found.', key_decisions='No key decisions found.',
                           questions='No open questions found.', rag_chain='test-chain')
        self.pipeline = Mock(return_value=self.result)
        main = types.ModuleType('main')
        main.run_pipeline = self.pipeline
        patcher = patch.dict(sys.modules, {'main': main})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(os.environ, {'GROQ_API_KEY': 'test', 'COHERE_API_KEY': 'test'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()

    def analyze(self):
        self.app.text_input[0].set_value('https://youtu.be/test')
        self.app.button[0].click().run()

    def test_initial_screen_and_bad_url(self):
        self.assertFalse(self.app.exception)
        self.app.text_input[0].set_value('https://github.com/example')
        self.app.button[0].click().run()
        self.assertTrue(self.app.error)
        self.pipeline.assert_not_called()

    def test_results_chat_and_clear_without_retranscribing(self):
        self.analyze()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.text_area[0].value, 'Full transcript')
        self.app.run()
        with patch('core.rag_engine.ask_question', return_value='An answer') as ask:
            self.app.chat_input[0].set_value('What happened?').run()
            ask.assert_called_once_with('test-chain', 'What happened?')
        self.pipeline.assert_called_once()
        self.assertEqual(len(self.app.session_state['messages']), 2)
        self.app.button[1].click().run()
        self.assertNotIn('result', self.app.session_state)
        self.assertFalse(self.app.exception)

    def test_analysis_failure_is_displayed(self):
        self.pipeline.side_effect = RuntimeError('test failure')
        self.analyze()
        self.assertTrue(self.app.error)
        self.assertFalse(self.app.exception)
        self.assertNotIn('result', self.app.session_state)
