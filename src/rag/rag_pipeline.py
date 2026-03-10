import os
import re
import json
import hashlib
from pathlib import Path

import chromadb
from google import genai
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

from src.resolver.company_resolver import CompanyResolver

load_dotenv()


class RAGPipeline:

    def __init__(
        self,
        chroma_dir="chroma_db",
        collection_name="nasdaq_docs",
        preferred_model="gemini-2.5-flash"
    ):

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        self.genai_client = genai.Client(api_key=api_key)
        self.model_name = preferred_model

        # ==============================
        # Resolve project root
        # ==============================

        project_root = Path(__file__).resolve().parents[2]

        self.raw_docs_dir = project_root / "data" / "raw_docs"
        self.chroma_dir = project_root / chroma_dir

        print("DEBUG project root:", project_root)
        print("DEBUG raw docs dir:", self.raw_docs_dir)

        # ==============================
        # Embedding
        # ==============================

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # ==============================
        # Chroma
        # ==============================

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # ==============================
        # Auto build vector DB if empty
        # ==============================

        self._initialize_collection_if_empty()

        self.collection_is_empty = self.collection.count() == 0

        print("DEBUG collection count:", self.collection.count())

        # ==============================

        self.resolver = CompanyResolver()

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.compare_patterns = [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\.?\b",
        ]

    # =========================================================
    # BUILD VECTOR DATABASE
    # =========================================================

    def _initialize_collection_if_empty(self):

        try:
            if self.collection.count() > 0:
                print("DEBUG collection already populated")
                return
        except:
            pass

        if not self.raw_docs_dir.exists():
            print("DEBUG raw_docs folder NOT FOUND")
            return

        txt_files = list(self.raw_docs_dir.glob("*.txt"))

        print("DEBUG txt files found:", len(txt_files))

        if len(txt_files) == 0:
            print("DEBUG no txt files")
            return

        batch_docs = []
        batch_ids = []
        batch_meta = []

        for file in txt_files:

            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except:
                continue

            chunks = self._chunk_text(text)

            meta = self._parse_metadata_from_filename(file)

            for i, chunk in enumerate(chunks):

                doc_id = hashlib.md5(
                    (file.name + str(i)).encode()
                ).hexdigest()

                batch_docs.append(chunk)
                batch_ids.append(doc_id)

                chunk_meta = meta.copy()
                chunk_meta["chunk"] = i

                batch_meta.append(chunk_meta)

        print("DEBUG adding chunks:", len(batch_docs))

        self.collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_meta
        )

        print("DEBUG build complete")

    # =========================================================

    def _chunk_text(self, text, size=1200, overlap=150):

        chunks = []

        start = 0

        while start < len(text):

            end = start + size

            chunk = text[start:end]

            chunks.append(chunk)

            start = end - overlap

        return chunks

    # =========================================================

    def _parse_metadata_from_filename(self, file):

        name = file.stem

        ticker = name.split("_")[0]

        doc_type = "unknown"
        source = "unknown"

        if "_yf_company_profile_" in name:
            doc_type = "yf_company_profile"
            source = "yahoo"

        if "_yf_financial_snapshot_" in name:
            doc_type = "yf_financial_snapshot"
            source = "yahoo"

        if "_yf_news_" in name:
            doc_type = "yf_news"
            source = "yahoo"

        if "_sec_companyfacts_" in name:
            doc_type = "sec_companyfacts"
            source = "sec"

        return {
            "ticker": ticker,
            "company": ticker,
            "doc_type": doc_type,
            "source": source,
            "title": name
        }

    # =========================================================
    # SEARCH
    # =========================================================

    def _raw_vector_search(self, query, n_results=5):

        if self.collection_is_empty:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        output = []

        for doc, meta in zip(docs, metas):

            output.append(
                {
                    "document": doc,
                    "metadata": meta
                }
            )

        return output

    # =========================================================

    def answer(self, query):

        retrieved = self._raw_vector_search(query)

        if len(retrieved) == 0:

            return {
                "answer":
                "The vector database is currently empty on this deployment. The app is running, but no documents have been loaded into Chroma yet.",
                "citations": []
            }

        context = "\n\n".join([x["document"] for x in retrieved])

        prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return {
            "answer": response.text,
            "citations": retrieved
        }