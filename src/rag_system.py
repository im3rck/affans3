"""Advanced RAG System with Hybrid Search and Reranking"""
import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest


class HybridSearchRAG:
    """RAG system with hybrid search (vector + BM25) and reranking"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "amazon_knowledge_base",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize reranker
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        # Storage for BM25
        self.documents = []
        self.bm25 = None

        # Try to get existing collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = None
            print(f"Collection {collection_name} does not exist yet")

    def _prepare_bm25(self, documents: List[Document]):
        """Prepare BM25 index from documents"""
        self.documents = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✓ BM25 index created with {len(documents)} documents")

    def index_documents(self, documents: List[Document]):
        """Index documents in both ChromaDB (vector) and BM25 (keyword)"""
        if not documents:
            print("No documents to index")
            return

        print(f"\nIndexing {len(documents)} documents...")

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"✓ Deleted existing collection")
        except Exception:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        print(f"✓ Indexed {len(documents)} documents in ChromaDB")

        # Prepare BM25 index
        self._prepare_bm25(documents)

    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Perform vector similarity search"""
        if self.collection is None:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Convert to documents with scores
        docs_with_scores = []
        if results['documents'][0]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1 - distance
                doc = Document(page_content=doc_text, metadata=metadata)
                docs_with_scores.append((doc, score))

        return docs_with_scores

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        if self.bm25 is None:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Return documents with scores
        docs_with_scores = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                docs_with_scores.append((self.documents[idx], float(scores[idx])))

        return docs_with_scores

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Perform hybrid search combining vector and BM25 search"""
        # Get results from both methods
        vector_results = self._vector_search(query, top_k=top_k * 2)
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # Normalize scores
        if vector_results:
            vector_scores = self._normalize_scores([score for _, score in vector_results])
            vector_results = [(doc, score) for (doc, _), score in zip(vector_results, vector_scores)]

        if bm25_results:
            bm25_scores = self._normalize_scores([score for _, score in bm25_results])
            bm25_results = [(doc, score) for (doc, _), score in zip(bm25_results, bm25_scores)]

        # Combine scores using weighted sum
        combined_scores = {}
        for doc, score in vector_results:
            doc_id = doc.page_content
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.vector_weight * score

        for doc, score in bm25_results:
            doc_id = doc.page_content
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.bm25_weight * score

        # Create document map
        doc_map = {}
        for doc, _ in vector_results + bm25_results:
            doc_map[doc.page_content] = doc

        # Sort by combined score
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k documents
        return [doc_map[doc_id] for doc_id, _ in sorted_docs[:top_k]]

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """Rerank documents using FlashRank"""
        if not documents:
            return []

        # Prepare passages for reranking
        passages = [{"text": doc.page_content} for doc in documents]

        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(rerank_request)

        # Sort by score and return top-k
        # FlashRank returns results with 'corpus_id' key, not 'index'
        reranked_docs = []
        for result in results[:top_k]:
            # Handle different possible key names from FlashRank
            idx = result.get('corpus_id', result.get('id', result.get('index')))
            if idx is not None:
                reranked_docs.append(documents[idx])

        return reranked_docs

    def retrieve(self, query: str, top_k: int = 5, rerank_top_k: int = 3) -> List[Document]:
        """
        Main retrieval method with hybrid search and reranking

        Args:
            query: Search query
            top_k: Number of documents to retrieve from hybrid search
            rerank_top_k: Number of documents to return after reranking

        Returns:
            List of top documents after reranking
        """
        # Hybrid search
        hybrid_results = self.hybrid_search(query, top_k=top_k)

        if not hybrid_results:
            return []

        # Rerank
        reranked_results = self.rerank_documents(query, hybrid_results, top_k=rerank_top_k)

        return reranked_results

    def get_context(self, query: str, top_k: int = 5, rerank_top_k: int = 3) -> str:
        """Get context string from retrieved documents"""
        documents = self.retrieve(query, top_k=top_k, rerank_top_k=rerank_top_k)

        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}\n")

        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the RAG system
    from document_processor import DocumentProcessor

    # Process documents
    processor = DocumentProcessor()
    pdf_files = [
        "HR & Benefits FAQ.pdf",
        "IT & Tech Support.pdf",
        "Workplace & Operations FAQ.pdf"
    ]
    csv_files = ["amazon.csv"]

    documents = processor.process_all_documents(pdf_files, csv_files)

    # Initialize and index
    rag = HybridSearchRAG()
    rag.index_documents(documents)

    # Test retrieval
    query = "How do I view my pay stub?"
    results = rag.retrieve(query, top_k=5, rerank_top_k=3)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:200]}...")
        print(f"   Source: {doc.metadata.get('filename', 'Unknown')}\n")
