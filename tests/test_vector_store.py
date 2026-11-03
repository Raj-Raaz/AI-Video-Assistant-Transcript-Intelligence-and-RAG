import unittest
from unittest.mock import patch
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from core import vector_store, rag_engine


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.5, 0.2] for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(':memory:')
        self.addCleanup(self.client.close)
        for name, value in [('get_qdrant_client', self.client), ('get_embeddings', LocalEmbeddings())]:
            patcher = patch.object(vector_store, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_first_run_creates_collection_and_indexes(self):
        self.assertFalse(self.client.collection_exists(vector_store.COLLECTION_NAME))
        vector_store.index_transcript('Nvidia builds accelerated computing platforms.', 'video-one')
        docs = vector_store.search_transcript('Nvidia', video_id='video-one')
        self.assertEqual(len(docs), 1)
        self.assertIn('Nvidia', docs[0].page_content)

    def test_repeat_index_does_not_duplicate_and_filter_isolates(self):
        for _ in range(2):
            vector_store.index_transcript('First video content', 'video-one')
        vector_store.index_transcript('Second video content', 'video-two')
        self.assertEqual(self.client.count(vector_store.COLLECTION_NAME).count, 2)
        docs = vector_store.search_transcript('content', video_id='video-one')
        self.assertEqual([d.page_content for d in docs], ['First video content'])

    def test_rag_indexes_current_transcript_and_uses_only_its_context(self):
        vector_store.index_transcript('Unrelated private video content', 'old-video')
        prompts = []
        def respond(prompt):
            prompts.append(prompt.to_string())
            return 'Answer'
        with patch.object(rag_engine, 'get_llm', return_value=RunnableLambda(respond)):
            chain = rag_engine.build_rag_chain('Current video discusses Nvidia and TPU.')
            self.assertEqual(chain.invoke('What is discussed?'), 'Answer')
        self.assertIn('Current video discusses Nvidia', prompts[0])
        self.assertNotIn('Unrelated private video', prompts[0])

    def test_empty_transcript_does_not_create_collection(self):
        with self.assertRaises(ValueError):
            rag_engine.build_rag_chain(' ')
        self.assertFalse(self.client.collection_exists(vector_store.COLLECTION_NAME))
