import os
import re
import json
import chromadb

from google import genai
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from src.resolver.company_resolver import CompanyResolver

load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: str = "chroma_db",
        collection_name: str = "nasdaq_docs",
        preferred_model: str = "gemini-2.5-flash",
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.genai_client = genai.Client(api_key=api_key)
        self.model_name = preferred_model

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        self.resolver = CompanyResolver()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 常見比較與問題詞
        self.compare_patterns = [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bwhich company\b",
            r"\bmore profitable\b",
            r"\bbetter than\b",
        ]

        self.stop_terms = {
            "tell", "me", "about", "please", "summarize", "summary", "show",
            "give", "what", "how", "is", "are", "the", "a", "an", "for",
            "of", "and", "or", "to", "in", "on", "with", "latest", "recent",
        }

        # query intent -> preferred doc types
        self.intent_doc_type_map = {
            "business": {"yf_company_profile", "sec_entity_profile"},
            "products": {"yf_company_profile", "sec_entity_profile"},
            "financial": {"yf_financial_snapshot", "sec_companyfacts"},
            "news": {"yf_news"},
            "filings": {"sec_recent_filings", "sec_entity_profile", "sec_companyfacts"},
            "comparison": {"yf_financial_snapshot", "sec_companyfacts", "yf_company_profile"},
        }

    # =========================================================
    # Helper: company handling
    # =========================================================
    def _is_comparison_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in self.compare_patterns)

    def _find_companies_in_query(self, query: str, current_company: dict | None = None) -> list[dict]:
        """
        找出 query 內涉及的公司。

        修正版重點：
        1. comparison query 不可太早 return
        2. resolve_many() 只要沒抓滿 2 家，就繼續往下掃 alias
        3. 單公司 query 才允許找到 1 家就結束
        """
        found = []
        seen = set()
        comparison_mode = self._is_comparison_query(query)

        # -----------------------------
        # 1) 優先用 resolve_many()
        # -----------------------------
        if hasattr(self.resolver, "resolve_many"):
            try:
                many = self.resolver.resolve_many(
                    query,
                    current_company=current_company,
                    max_companies=2 if comparison_mode else 1,
                )
                for item in many:
                    if item and item.get("ticker") and item["ticker"] not in seen:
                        seen.add(item["ticker"])
                        found.append(item)

                # 單公司模式：找到 1 家就可以先結束
                if not comparison_mode and len(found) >= 1:
                    return found[:1]

                # 比較模式：只有找到 2 家才可以結束
                if comparison_mode and len(found) >= 2:
                    return found[:2]

            except Exception:
                pass

        # -----------------------------
        # 2) exact ticker scan
        # -----------------------------
        ticker_set = getattr(self.resolver, "ticker_set", set())
        ticker_map = getattr(self.resolver, "ticker_map", {})

        tokens = re.findall(r"\$?[A-Za-z]{1,10}\b", query)
        for token in tokens:
            raw = token[1:] if token.startswith("$") else token

            if raw.isalpha() and raw.isupper() and raw in ticker_set:
                payload = ticker_map.get(raw)
                if payload and payload["ticker"] not in seen:
                    seen.add(payload["ticker"])
                    found.append(payload)

        if not comparison_mode and len(found) >= 1:
            return found[:1]
        if comparison_mode and len(found) >= 2:
            return found[:2]

        # -----------------------------
        # 3) alias scan from alias_map
        # -----------------------------
        alias_map = getattr(self.resolver, "alias_map", None)

        if isinstance(alias_map, dict):
            q_norm = query.lower().strip()
            matched_aliases = []

            for alias, payload in alias_map.items():
                if not alias or len(alias) < 2:
                    continue

                pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
                if re.search(pattern, q_norm):
                    matched_aliases.append((len(alias), payload, alias))

            # 長 alias 優先，避免 "apple" 被更短或更泛 alias 干擾
            matched_aliases = sorted(matched_aliases, key=lambda x: x[0], reverse=True)

            for _, payload, alias in matched_aliases:
                ticker = payload.get("ticker")
                if ticker and ticker not in seen:
                    seen.add(ticker)
                    found.append({
                        "ticker": payload.get("ticker"),
                        "company": payload.get("company"),
                    })

                if not comparison_mode and len(found) >= 1:
                    return found[:1]
                if comparison_mode and len(found) >= 2:
                    return found[:2]

        # -----------------------------
        # 4) comparison query 額外 fallback：
        #    從 "X and Y" / "X vs Y" 這類模式拆字面公司名
        # -----------------------------
        if comparison_mode:
            q_clean = re.sub(r"[?.!,]", " ", query)
            q_clean = re.sub(r"\s+", " ", q_clean).strip()

            # 常見比較結構
            patterns = [
                r"compare\s+(.+?)\s+and\s+(.+)",
                r"compare\s+(.+?)\s+vs\.?\s+(.+)",
                r"(.+?)\s+vs\.?\s+(.+)",
                r"(.+?)\s+versus\s+(.+)",
            ]

            for p in patterns:
                m = re.search(p, q_clean, flags=re.IGNORECASE)
                if m:
                    left = m.group(1).strip()
                    right = m.group(2).strip()

                    # 清掉尾部功能詞
                    def normalize_candidate(x: str) -> str:
                        x = re.sub(
                            r"\b(in terms of|financial performance|business and financial performance|profitability|performance)\b",
                            "",
                            x,
                            flags=re.IGNORECASE,
                        )
                        x = re.sub(r"\s+", " ", x).strip(" ,.")
                        return x

                    candidates = [normalize_candidate(left), normalize_candidate(right)]

                    alias_map = getattr(self.resolver, "alias_map", {})
                    alias_list = getattr(self.resolver, "alias_list", [])

                    for cand in candidates:
                        cand_norm = cand.lower()

                        # exact alias first
                        if cand_norm in alias_map:
                            payload = alias_map[cand_norm]
                            if payload["ticker"] not in seen:
                                seen.add(payload["ticker"])
                                found.append({
                                    "ticker": payload.get("ticker"),
                                    "company": payload.get("company"),
                                })
                            continue

                        # fuzzy fallback
                        try:
                            best = process.extractOne(cand_norm, alias_list, scorer=fuzz.WRatio)
                            if best and best[1] >= 90:
                                payload = alias_map[best[0]]
                                if payload["ticker"] not in seen:
                                    seen.add(payload["ticker"])
                                    found.append({
                                        "ticker": payload.get("ticker"),
                                        "company": payload.get("company"),
                                    })
                        except Exception:
                            pass

                    break

        # -----------------------------
        # 5) 最後處理 fallback
        # -----------------------------
        if comparison_mode:
            return found[:2]

        if not found and current_company:
            found.append(current_company)

        return found[:1]

    def _strip_company_mentions(self, query: str, company_context: dict | None) -> str:
        if not company_context:
            q = re.sub(r"\s+", " ", query).strip(" ?,.")
            return q or "company profile business financial condition"

        q = query
        ticker = company_context.get("ticker", "")
        company = company_context.get("company", "")

        patterns = []
        if ticker:
            patterns.append(rf"(?<!\w){re.escape(ticker)}(?!\w)")
        if company:
            patterns.append(rf"(?<!\w){re.escape(company)}(?!\w)")
            first_word = company.split()[0]
            patterns.append(rf"(?<!\w){re.escape(first_word)}(?!\w)")

        for p in patterns:
            q = re.sub(p, " ", q, flags=re.IGNORECASE)

        q = re.sub(r"\s+", " ", q).strip(" ?,.")
        return q or "company profile business financial condition"
    
    def _infer_intents(self, query: str) -> set[str]:
        q = query.lower()
        intents = set()

        if any(x in q for x in ["business model", "business", "overview", "company", "products", "services", "core products", "segments"]):
            intents.add("business")

        if any(x in q for x in ["products", "services", "core products"]):
            intents.add("products")

        if any(x in q for x in [
            "revenue", "net income", "profit", "profitability", "financial",
            "financial condition", "cash flow", "assets", "liabilities",
            "balance sheet", "margin", "operating cash flow", "earnings"
        ]):
            intents.add("financial")

        if any(x in q for x in ["news", "latest", "recent developments", "recent news", "what is happening"]):
            intents.add("news")

        if any(x in q for x in ["filing", "filings", "10-k", "10-q", "sec"]):
            intents.add("filings")

        if self._is_comparison_query(query):
            intents.add("comparison")

        if not intents:
            intents.add("business")

        return intents

    def _safe_parse_queries_from_llm(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()

        # remove markdown fences if any
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # try direct json
        try:
            data = json.loads(text)
            queries = data.get("queries", [])
            return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        except Exception:
            pass

        # try extracting first JSON object
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                queries = data.get("queries", [])
                return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
            except Exception:
                pass

        # fallback: line-based parsing
        lines = [x.strip("-• \n\t") for x in text.splitlines() if x.strip()]
        lines = [x for x in lines if len(x) > 2]
        return lines[:3]
    
    

    # =========================================================
    # Helper: query expansion
    # =========================================================
    def _rule_based_query_expansion(
        self,
        user_query: str,
        company_context: dict | None,
    ) -> list[str]:
        """
        規則式 query expansion：
        針對 business / financial / news / filings / comparison 類問題
        補足與文件 wording 更接近的查詢。
        """
        q_lower = user_query.lower()
        expansions = []

        focus_query = self._strip_company_mentions(user_query, company_context)

        # 通用基底
        expansions.extend([
            focus_query,
            "company profile",
            "business overview",
        ])

        # Business / products
        if any(x in q_lower for x in ["business model", "business", "products", "services", "core products"]):
            expansions.extend([
                "company profile",
                "business overview",
                "products and services",
                "long business summary",
                "business segments",
                "core products",
            ])

        # Financials
        if any(x in q_lower for x in [
            "revenue", "net income", "profit", "profitability", "financial",
            "financial condition", "cash flow", "assets", "liabilities",
            "balance sheet", "margin", "operating cash flow"
        ]):
            expansions.extend([
                "financial snapshot",
                "sec company facts",
                "revenue net income assets liabilities equity cash flow",
                "financial performance",
                "balance sheet",
                "cash flow statement",
                "profit margins",
            ])

        # News
        if any(x in q_lower for x in ["news", "latest", "recent developments", "recent news", "what is happening"]):
            expansions.extend([
                "latest news",
                "recent news",
                "recent developments",
                "news digest",
                "press release",
            ])

        # Filings
        if any(x in q_lower for x in ["filing", "filings", "10-k", "10-q", "sec"]):
            expansions.extend([
                "recent sec filings",
                "10-k 10-q filings",
                "sec filings",
                "sec entity profile",
                "sec company facts",
            ])

        # Comparison
        if self._is_comparison_query(user_query):
            expansions.extend([
                "financial performance comparison",
                "revenue profitability",
                "business overview comparison",
                "market snapshot",
            ])

        # 去掉太空泛字詞
        cleaned = []
        seen = set()

        for q in expansions:
            q = re.sub(r"\s+", " ", q).strip()
            if not q:
                continue
            q_norm = q.lower()
            if q_norm not in seen:
                seen.add(q_norm)
                cleaned.append(q)

        return cleaned[:12]

    
    
    
    
    def _llm_query_rewrites(self, user_query: str, company_context: dict | None) -> list[str]:
        """
        用 Gemini 將自然語言問題改寫成較適合 retrieval 的短 query。
        若失敗則回空陣列。
        """
        company_text = ""
        if company_context:
            company_text = f"Target company: {company_context['company']} ({company_context['ticker']})"

        prompt = f"""
    Rewrite the user's question into 3 short retrieval queries.

    Rules:
    - Keep each query short and retrieval-friendly.
    - Prefer document-oriented wording like:
    company profile, business overview, products and services,
    revenue, net income, financial snapshot, SEC filings, recent news.
    - If a target company is given, keep all queries tied to that company.
    - Return ONLY valid JSON in this format:
    {{"queries": ["q1", "q2", "q3"]}}

    {company_text}

    User question:
    {user_query}
    """.strip()

        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = response.text.strip() if hasattr(response, "text") else str(response)
            queries = self._safe_parse_queries_from_llm(text)
            return queries[:3]
        except Exception:
            return []

    def generate_queries(self, user_query: str, company_context: dict | None) -> list[str]:
        """
        Multi-query retrieval:
        1. 原始 query
        2. 去除公司名稱後的 focus query
        3. rule-based query expansion
        4. deterministic company-aware variants
        5. Gemini rewrites
        """
        focus_query = self._strip_company_mentions(user_query, company_context)
        intents = self._infer_intents(user_query)

        queries = [user_query.strip(), focus_query]

        # 規則式 expansion：文件導向 wording
        if "business" in intents or "products" in intents:
            queries.extend([
                "company profile",
                "business overview",
                "products and services",
                "long business summary",
                "business segments",
            ])

        if "financial" in intents:
            queries.extend([
                "financial snapshot",
                "sec company facts",
                "revenue net income assets liabilities equity cash flow",
                "financial performance",
                "balance sheet",
                "profit margins",
            ])

        if "news" in intents:
            queries.extend([
                "latest news",
                "recent news",
                "recent developments",
                "news digest",
            ])

        if "filings" in intents:
            queries.extend([
                "recent sec filings",
                "10-k 10-q filings",
                "sec filings",
                "sec entity profile",
                "sec company facts",
            ])

        if "comparison" in intents:
            queries.extend([
                "financial performance comparison",
                "revenue profitability",
                "business overview comparison",
                "market snapshot",
            ])

        # 公司導向 deterministic variants
        if company_context:
            ticker = company_context["ticker"]
            company = company_context["company"]

            queries.extend([
                f"{company} {focus_query}",
                f"{ticker} {focus_query}",
                f"{company} company profile business overview",
                f"{company} products and services",
                f"{company} financial snapshot sec company facts",
                f"{company} recent news",
                f"{company} sec filings",
            ])
        else:
            queries.extend([
                f"{focus_query} company profile",
                f"{focus_query} business overview",
                f"{focus_query} financial snapshot",
                f"{focus_query} recent news",
            ])

        # LLM rewrites
        queries.extend(self._llm_query_rewrites(user_query, company_context))

        cleaned = []
        seen = set()
        for q in queries:
            q = re.sub(r"\s+", " ", q).strip()
            q_norm = q.lower()
            if q_norm and q_norm not in seen:
                seen.add(q_norm)
                cleaned.append(q)

        return cleaned[:15]

    # =========================================================
    # Helper: vector retrieval
    # =========================================================
    def _raw_vector_search(
        self,
        query: str,
        n_results: int = 20,
        company_context: dict | None = None,
    ) -> list[dict]:
        where_filter = None
        if company_context:
            where_filter = {"ticker": company_context["ticker"]}

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)

        output = []
        for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
            output.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return output

    def _deduplicate_candidates(self, candidates: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for item in candidates:
            doc_id = item["id"]
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(item)
        return unique

    def _hybrid_rank(
        self,
        query: str,
        candidates: list[dict],
        final_k: int = 5,
        intents: set[str] | None = None,
    ) -> list[dict]:
        if not candidates:
            return []

        intents = intents or self._infer_intents(query)

        docs = [c["document"] for c in candidates]
        tokenized_docs = [re.findall(r"\w+", d.lower()) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))

        for c, s in zip(candidates, bm25_scores):
            c["bm25_score"] = float(s)

        preferred_doc_types = set()
        for intent in intents:
            preferred_doc_types.update(self.intent_doc_type_map.get(intent, set()))

        def doc_type_boost(x):
            doc_type = x["metadata"].get("doc_type")
            return 0.2 if doc_type in preferred_doc_types else 0.0

        def combined_score(x):
            dist = x["distance"] if x["distance"] is not None else 1.0
            semantic_score = 1 / (1 + dist)
            return 0.55 * semantic_score + 0.25 * x["bm25_score"] + doc_type_boost(x)

        coarse = sorted(candidates, key=combined_score, reverse=True)[:25]

        pairs = [(query, c["document"]) for c in coarse]
        rerank_scores = self.reranker.predict(pairs)

        for c, s in zip(coarse, rerank_scores):
            c["rerank_score"] = float(s)

        ranked = sorted(coarse, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:final_k]

    def hybrid_search(self, query: str, company_context: dict | None, final_k: int = 5) -> list[dict]:
        """
        針對單一公司或單一上下文做 query expansion + multi-query retrieval + hybrid rank。
        這版會同時做：
        - company filtered retrieval
        - partial global retrieval
        避免「有結果但都是爛 chunk」。
        """
        query_variants = self.generate_queries(query, company_context)
        intents = self._infer_intents(query)

        all_candidates = []

        # A. 公司限定 retrieval
        if company_context:
            for q in query_variants:
                batch = self._raw_vector_search(q, n_results=20, company_context=company_context)
                all_candidates.extend(batch)

        # B. 全域 retrieval（不再只在 empty 時才做）
        #    只對前幾個最重要 query 做，避免 noise 太高
        global_queries = query_variants[:6]
        for q in global_queries:
            batch = self._raw_vector_search(q, n_results=10, company_context=None)
            all_candidates.extend(batch)

        candidates = self._deduplicate_candidates(all_candidates)

        return self._hybrid_rank(
            query=query,
            candidates=candidates,
            final_k=final_k,
            intents=intents,
        )

    # =========================================================
    # Comparison / multi-company retrieval
    # =========================================================
    def _comparison_search(self, query: str, companies: list[dict], final_k: int = 5) -> list[dict]:
        """
        comparison query:
        - 各公司各自 retrieve
        - 再加 global retrieval
        - 合併後 rerank
        """
        merged = []

        for company_context in companies[:2]:
            partial = self.hybrid_search(
                query=query,
                company_context=company_context,
                final_k=max(final_k, 6),
            )
            merged.extend(partial)

        # 補一點 global retrieval，避免某家公司文件沒被 alias / filter 命中
        for q in self.generate_queries(query, None)[:6]:
            merged.extend(self._raw_vector_search(q, n_results=10, company_context=None))

        merged = self._deduplicate_candidates(merged)

        return self._hybrid_rank(
            query=query,
            candidates=merged,
            final_k=final_k,
            intents={"comparison", "financial", "business"},
        )

    # =========================================================
    # Prompting / generation
    # =========================================================
    def build_prompt(
        self,
        user_query: str,
        retrieved_chunks: list[dict],
        conversation_history: list[tuple[str, str]] | None,
        company_context: dict | None,
    ) -> str:
        history_text = ""
        if conversation_history:
            trimmed = conversation_history[-10:]
            history_text = "\n".join([f"{role.upper()}: {msg}" for role, msg in trimmed])

        company_text = ""
        if company_context:
            company_text = f"Resolved company: {company_context['company']} ({company_context['ticker']})"

        context_blocks = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            meta = item["metadata"]
            context_blocks.append(
                f"[S{idx}]\n"
                f"Ticker: {meta.get('ticker')}\n"
                f"Company: {meta.get('company')}\n"
                f"Source: {meta.get('source')}\n"
                f"Doc Type: {meta.get('doc_type')}\n"
                f"Title: {meta.get('title')}\n"
                f"Content:\n{item['document']}\n"
            )

        context_text = "\n\n".join(context_blocks)

        return f"""
You are a grounded financial chatbot for Nasdaq listed companies.

Rules:
1. Answer ONLY from retrieved context.
2. Never invent facts, numbers, products, events, or assumptions.
3. If evidence is insufficient, say exactly:
   "I don't have enough information in the retrieved documents to answer that."
4. Use inline citations like [S1], [S2].
5. Use conversation history only to understand follow-up questions, not as factual evidence.
6. If the question is comparative, compare only what is supported by the retrieved context.
7. Keep the answer factual, structured, and concise.

{company_text}

Conversation history:
{history_text}

Retrieved context:
{context_text}

User question:
{user_query}

Write:
- Direct answer
- Optional bullet summary
- Final line: Sources Used: [S1], [S2], ...
""".strip()

    def _generate_answer(self, prompt: str) -> str:
        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()

    # =========================================================
    # Main answer
    # =========================================================
    def answer(
        self,
        query: str,
        conversation_history: list[tuple[str, str]] | None = None,
        current_company: dict | None = None,
    ) -> dict:
        companies = self._find_companies_in_query(query, current_company=current_company)
        print("DEBUG companies found:", companies)

        # 單公司 / 無公司
        if not self._is_comparison_query(query):
            company_context = companies[0] if companies else None
            retrieved = self.hybrid_search(
                query=query,
                company_context=company_context,
                final_k=5,
            )
        else:
            company_context = companies[0] if companies else None

            if len(companies) >= 2:
                retrieved = self._comparison_search(
                    query=query,
                    companies=companies,
                    final_k=5,
                )
            else:
                print("DEBUG comparison fallback triggered; companies found:", companies)
                retrieved = self.hybrid_search(
                    query=query,
                    company_context=None,
                    final_k=5,
                )

        if not retrieved:
            return {
                "answer": "I couldn't retrieve any relevant documents from the vector database.",
                "citations": [],
                "company_context": company_context,
                "model_name": self.model_name,
            }

        prompt = self.build_prompt(
            user_query=query,
            retrieved_chunks=retrieved,
            conversation_history=conversation_history,
            company_context=company_context,
        )

        answer_text = self._generate_answer(prompt)

        citations = []
        for idx, item in enumerate(retrieved, start=1):
            meta = item["metadata"]
            citations.append(
                {
                    "label": f"S{idx}",
                    "ticker": meta.get("ticker"),
                    "company": meta.get("company"),
                    "source": meta.get("source"),
                    "doc_type": meta.get("doc_type"),
                    "title": meta.get("title"),
                    "snippet": item["document"][:320] + ("..." if len(item["document"]) > 320 else ""),
                }
            )

        return {
            "answer": answer_text,
            "citations": citations,
            "company_context": company_context,
            "model_name": self.model_name,
        }