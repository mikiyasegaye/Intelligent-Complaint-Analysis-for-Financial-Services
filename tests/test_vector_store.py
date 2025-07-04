"""
Unit tests for the vector store module.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import shutil

from src.models.vector_store import VectorStore
from src.models.text_processor import TextProcessor


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            persist_directory=self.temp_dir,
            collection_name='test_collection'
        )

        # Create test data
        self.processor = TextProcessor(device='cpu')
        self.test_text = "This is a test complaint about a banking issue."
        self.test_metadata = {
            'complaint_id': 'test123',
            'product': 'banking',
            'company': 'test bank'
        }

        # Generate test embeddings
        chunks = self.processor.create_chunks(
            self.test_text, self.test_metadata)
        self.ids, self.metadatas, self.embeddings = self.processor.generate_embeddings(
            chunks,
            show_progress=False
        )
        self.documents = [chunk.page_content for chunk in chunks]

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_add_documents(self):
        """Test adding documents to the vector store."""
        initial_count = self.vector_store.count

        # Add documents
        self.vector_store.add_documents(
            self.ids,
            self.embeddings,
            self.metadatas,
            self.documents
        )

        # Verify documents were added
        self.assertEqual(
            self.vector_store.count,
            initial_count + len(self.documents)
        )

    def test_query(self):
        """Test querying the vector store."""
        # Add test documents
        self.vector_store.add_documents(
            self.ids,
            self.embeddings,
            self.metadatas,
            self.documents
        )

        # Create test query
        query_text = "banking issue complaint"
        query_embedding = self.processor.embedding_model.encode(
            [query_text],
            convert_to_numpy=True
        )[0]

        # Test basic query
        results = self.vector_store.query(
            query_embedding,
            n_results=1
        )

        # Check result structure
        self.assertIn('documents', results)
        self.assertIn('metadatas', results)
        self.assertIn('distances', results)

        # Check result content
        self.assertEqual(len(results['documents'][0]), 1)
        self.assertEqual(len(results['metadatas'][0]), 1)
        self.assertEqual(len(results['distances'][0]), 1)

    def test_peek(self):
        """Test peeking at documents in the store."""
        # Add test documents
        self.vector_store.add_documents(
            self.ids,
            self.embeddings,
            self.metadatas,
            self.documents
        )

        # Test peek
        results = self.vector_store.peek(n=1)

        # Check result structure
        self.assertIn('ids', results)
        self.assertIn('embeddings', results)
        self.assertIn('documents', results)
        self.assertIn('metadatas', results)


if __name__ == '__main__':
    unittest.main()
