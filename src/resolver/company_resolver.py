import os
import re
import pandas as pd
from rapidfuzz import process, fuzz


COMMON_WORDS = {
    "can", "tell", "show", "what", "how", "why", "when", "where", "who",
    "please", "summarize", "summary", "about", "give", "find", "is", "are",
    "the", "a", "an", "and", "or", "of", "for", "to", "me", "you", "its",
    "latest", "recent", "news", "compare", "comparison", "versus", "vs",
    "financial", "condition", "performance", "business", "model", "products",
    "services", "revenue", "profit", "profitability", "net", "income",
    "assets", "liabilities", "cash", "flow", "with", "based", "on", "from",
    "related", "overview", "explain", "describe", "latest", "current"
}


LEGAL_SUFFIX_PATTERN = (
    r"\b("
    r"inc|inc\.|corp|corp\.|corporation|ltd|ltd\.|plc|holdings|holding|"
    r"group|company|co|co\.|limited|sa|ag|nv|lp|llc|adr|ads|class"
    r")\b"
)


class CompanyResolver:
    def __init__(self, csv_path: str = "data/companies.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Company master not found: {csv_path}. Run build_index.py first."
            )

        self.df = pd.read_csv(csv_path)
        self.df["ticker"] = self.df["ticker"].astype(str).str.upper().str.strip()
        self.df["company"] = self.df["company"].astype(str).str.strip()

        self.df = self.df.dropna(subset=["ticker", "company"]).reset_index(drop=True)

        self.ticker_set = set(self.df["ticker"].tolist())

        # ticker -> payload
        self.ticker_map = {}
        # alias -> payload
        self.alias_map = {}
        self.alias_list = []

        for _, row in self.df.iterrows():
            ticker = row["ticker"]
            company = row["company"]

            payload = {
                "ticker": ticker,
                "company": company,
            }

            self.ticker_map[ticker] = payload

            aliases = self._build_aliases(company=company, ticker=ticker)

            for alias in aliases:
                alias = alias.strip()
                if alias and alias not in self.alias_map:
                    self.alias_map[alias] = payload
                    self.alias_list.append(alias)

        # 依 alias 長度排序，長的優先，避免 "apple" 與 "apple inc" 互相干擾
        self.alias_list = sorted(set(self.alias_list), key=len, reverse=True)

    # =========================================================
    # Alias construction
    # =========================================================
    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[,/&()\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _strip_legal_suffixes(self, text: str) -> str:
        text = re.sub(LEGAL_SUFFIX_PATTERN, " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_aliases(self, company: str, ticker: str) -> set[str]:
        """
        產生同一家公司可接受的 alias。
        """
        aliases = set()

        company_norm = self._normalize_text(company)
        simplified = self._strip_legal_suffixes(company_norm)
        simplified = self._normalize_text(simplified)

        aliases.add(ticker.lower())
        aliases.add(company_norm)

        if simplified:
            aliases.add(simplified)

        # 前 1 / 2 / 3 個詞
        words = simplified.split()
        if len(words) >= 1 and words[0] not in COMMON_WORDS:
            aliases.add(words[0])

        if len(words) >= 2:
            first_two = " ".join(words[:2]).strip()
            if first_two:
                aliases.add(first_two)

        if len(words) >= 3:
            first_three = " ".join(words[:3]).strip()
            if first_three:
                aliases.add(first_three)

        # 常見去尾碼版本
        company_no_punct = re.sub(r"[^a-zA-Z0-9\s]", " ", company_norm)
        company_no_punct = re.sub(r"\s+", " ", company_no_punct).strip()
        if company_no_punct:
            aliases.add(company_no_punct)

        # 去除很短或太泛的 alias
        cleaned = set()
        for a in aliases:
            a = a.strip()
            if not a:
                continue
            if len(a) == 1:
                continue
            if a in COMMON_WORDS:
                continue
            cleaned.add(a)

        return cleaned

    # =========================================================
    # Matchers
    # =========================================================
    def _find_exact_ticker(self, query: str) -> dict | None:
        """
        只把真正像 ticker 的 token 當 ticker：
        - 全大寫 AAPL
        - 或 $AAPL
        不把句首 Can / Tell 這種普通單字誤判成 ticker。
        """
        tokens = re.findall(r"\$?[A-Za-z]{1,5}\b", query)
        for token in tokens:
            raw = token[1:] if token.startswith("$") else token

            if raw.isalpha() and raw.isupper() and raw in self.ticker_set:
                return self.ticker_map[raw]

        return None

    def _find_all_exact_aliases(self, query: str) -> list[dict]:
        q = self._normalize_text(query)

        matches = []
        seen_tickers = set()

        for alias in self.alias_list:
            # 用 lookaround 保證完整詞匹配
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, q):
                payload = self.alias_map[alias]
                ticker = payload["ticker"]
                if ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    matches.append(payload)

        return matches

    def _find_fuzzy_alias(self, query: str) -> dict | None:
        q = self._normalize_text(query)

        # 先去掉常見問題詞
        q_tokens = [t for t in re.findall(r"[a-zA-Z]+", q) if t not in COMMON_WORDS]
        q_clean = " ".join(q_tokens).strip()

        if not q_clean:
            return None

        best = process.extractOne(
            q_clean,
            self.alias_list,
            scorer=fuzz.WRatio,
        )

        if best and best[1] >= 92:
            return self.alias_map[best[0]]

        return None

    # =========================================================
    # Public methods
    # =========================================================
    def resolve(self, query: str, current_company: dict | None = None) -> dict | None:
        """
        回傳單一最可能公司。
        """
        if not query or not query.strip():
            return current_company

        # 1) 先抓真正大寫 ticker
        match = self._find_exact_ticker(query)
        if match:
            return match

        # 2) exact alias
        exact_matches = self._find_all_exact_aliases(query)
        if exact_matches:
            return exact_matches[0]

        # 3) fuzzy alias
        match = self._find_fuzzy_alias(query)
        if match:
            return match

        # 4) 若沒有新公司，但有既有上下文，沿用目前公司
        return current_company

    def resolve_many(self, query: str, current_company: dict | None = None, max_companies: int = 2) -> list[dict]:
        """
        給 comparison / multi-company query 用。
        先抓 ticker，再抓 exact aliases；若都沒有，才 fuzzy。
        """
        if not query or not query.strip():
            return [current_company] if current_company else []

        results = []
        seen_tickers = set()

        # 1) exact tickers
        tokens = re.findall(r"\$?[A-Za-z]{1,5}\b", query)
        for token in tokens:
            raw = token[1:] if token.startswith("$") else token
            if raw.isalpha() and raw.isupper() and raw in self.ticker_set:
                payload = self.ticker_map[raw]
                if payload["ticker"] not in seen_tickers:
                    seen_tickers.add(payload["ticker"])
                    results.append(payload)

        # 2) exact aliases
        exact_matches = self._find_all_exact_aliases(query)
        for payload in exact_matches:
            if payload["ticker"] not in seen_tickers:
                seen_tickers.add(payload["ticker"])
                results.append(payload)

        # 3) 若完全沒抓到，試 fuzzy
        if not results:
            fuzzy = self._find_fuzzy_alias(query)
            if fuzzy:
                results.append(fuzzy)

        # 4) 若還是沒有，沿用上下文
        if not results and current_company:
            results.append(current_company)

        return results[:max_companies]