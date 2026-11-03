import ast
import unittest
from pathlib import Path
from unittest.mock import Mock

from core.llm_utils import RateLimitError


class PipelineTests(unittest.TestCase):
    def test_each_run_processes_audio_and_transcribes_again(self):
        source = Path('main.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'run_pipeline')
        process = Mock(return_value=['audio'])
        transcribe = Mock(return_value='transcript')
        scope = dict(process_input_audio=process, transcribe_audio_chunks=transcribe,
                     generate_title=Mock(side_effect=RateLimitError('limited')))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), scope)
        for _ in range(2):
            with self.assertRaises(RateLimitError):
                scope['run_pipeline']('https://example.com/video', 'en')
        self.assertEqual(process.call_count, 2)
        self.assertEqual(transcribe.call_count, 2)

    def test_pipeline_uses_one_extraction_call(self):
        source = Path('main.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'run_pipeline')
        information = {"action_items": "tasks", "key_decisions": "decisions", "questions": "questions"}
        extract = Mock(return_value=information)
        scope = dict(process_input_audio=Mock(return_value=['audio']),
                     transcribe_audio_chunks=Mock(return_value='transcript'),
                     generate_title=Mock(return_value='title'),
                     summarize_text=Mock(return_value='summary'),
                     extract_information=extract, build_rag_chain=Mock(return_value='chat'))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), scope)
        result = scope['run_pipeline']('video', 'en')
        extract.assert_called_once_with('transcript')
        for field, value in information.items():
            self.assertEqual(result[field], value)
