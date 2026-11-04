import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from core import transcriber, rag_engine
import test_vector_store


class TranscriptionTimestampTests(unittest.TestCase):
    def test_whisper_preserves_segment_times(self):
        model = Mock()
        model.transcribe.return_value = {"text": " Hello", "segments": [
            {"text": " Hello", "start": 1.2, "end": 3.4}]}
        with patch.object(transcriber, 'load_model', return_value=model):
            result = transcriber.transcribe_audio_whisper('audio.wav', with_segments=True)
        self.assertEqual(result['segments'], [{'text': 'Hello', 'start': 1.2, 'end': 3.4}])

    def test_offsets_use_actual_audio_duration(self):
        results = [
            {'text': 'First', 'segments': [{'text': 'First', 'start': 2, 'end': 5}]},
            {'text': 'Second', 'segments': [{'text': 'Second', 'start': 1, 'end': 3}]},
        ]
        # Simulate a 10-minute first chunk and a short final chunk.
        with patch.object(transcriber, 'transcribe_audio_whisper', side_effect=results), \
             patch.object(transcriber.AudioSegment, 'from_file', side_effect=[range(600000), range(4000)]):
            result = transcriber.transcribe_audio_chunks(['a', 'b'], with_segments=True)
        self.assertEqual(result['text'], 'First Second')
        self.assertEqual(result['segments'][1]['start'], 601)
        self.assertEqual(result['segments'][1]['end'], 603)

    def test_sarvam_uses_piece_windows_and_partial_final_piece(self):
        from pydub import AudioSegment
        audio = AudioSegment.silent(duration=26000)
        with patch.object(transcriber, 'SARVAM_API_KEY', 'test'), \
             patch.object(transcriber.AudioSegment, 'from_wav', return_value=audio), \
             patch.object(transcriber.AudioSegment, 'export'), \
             patch.object(transcriber, 'send_to_sarvam', side_effect=['First', 'Second']):
            result = transcriber.transcribe_audio_sarvam('test.wav', with_segments=True)
        self.assertEqual(result['segments'][1], {'text': 'Second', 'start': 25, 'end': 26})


class TimestampRetrievalTests(unittest.TestCase):
    setUp = test_vector_store.VectorStoreTests.setUp

    def test_timed_rag_returns_grounded_link_and_stores_metadata(self):
        segments = [{'text': 'A useful fact.', 'start': 601.5, 'end': 606.2}]
        with patch.object(rag_engine, 'get_llm', return_value=RunnableLambda(lambda _: 'The fact.')):
            chain = rag_engine.build_rag_chain('A useful fact.', segments, 'https://youtu.be/yB9SC4HbH5o?si=abc')
            answer = chain.invoke('What fact?')
        self.assertIn('[10:01–10:07](https://www.youtube.com/watch?v=yB9SC4HbH5o&t=601s)', answer)
        points, _ = self.client.scroll('video_transcripts')
        self.assertEqual(points[0].payload['metadata']['start_seconds'], 601.5)

    def test_overlapping_text_chunks_preserve_repeated_segment_mapping(self):
        from core import vector_store
        segments = [{'text': 'Repeated speech.', 'start': i * 2, 'end': i * 2 + 1} for i in range(100)]
        transcript = ' '.join(s['text'] for s in segments)
        vector_store.index_transcript(transcript, 'timed', segments=segments)
        points, _ = self.client.scroll('video_transcripts', limit=100)
        metadata = sorted([p.payload['metadata'] for p in points], key=lambda m: m['chunk_index'])
        self.assertGreater(len(metadata), 1)
        self.assertEqual(metadata[0]['start_seconds'], 0)
        self.assertGreater(metadata[1]['start_seconds'], 0)
        self.assertEqual(metadata[-1]['end_seconds'], 199)


class CitationFormattingTests(unittest.TestCase):
    def test_upload_sources_are_plain_times_and_deduplicated(self):
        doc = Document(page_content='text', metadata={'start_seconds': 3601, 'end_seconds': 3605})
        output = rag_engine.source_citations([doc, doc], r'C:\private\upload.mp4')
        self.assertEqual(output.count('1:00:01–1:00:05'), 1)
        self.assertNotIn('private', output)
        self.assertNotIn('http', output)

    def test_missing_timestamps_do_not_invent_sources(self):
        self.assertEqual(rag_engine.source_citations([Document(page_content='text')], None), '')

    def test_youtube_url_variants_and_untrusted_hosts(self):
        for url in ['https://youtube.com/watch?v=yB9SC4HbH5o&t=30',
                    'https://youtube.com/shorts/yB9SC4HbH5o',
                    'https://youtu.be/yB9SC4HbH5o']:
            self.assertEqual(rag_engine.youtube_url(url), 'https://www.youtube.com/watch?v=yB9SC4HbH5o')
        self.assertIsNone(rag_engine.youtube_url('https://youtube.com.evil.test/watch?v=yB9SC4HbH5o'))
