# Architecture Decision Records (ADRs)

## ADR-001: Target Exchanges Selection (HKEX & KRX)
- **Status**: Proposed / Awaiting Final User Confirmation
- **Context**: The assignment document is titled *"HengSeng Sentiment Price Prediction agent"*, but lists local social sources from both Hong Kong (LIHKG, Futu, AAStocks) and South Korea (Naver Finance 종목토론실, Daum Finance, Blind, KakaoTalk).
- **Decision**: Explicitly target **Hong Kong Stock Exchange (HKEX / Hang Seng)** as Exchange 1, and **Korea Exchange (KRX / KOSPI)** as Exchange 2.
- **Consequences**:
  - Requires handling both Hong Kong Dollar (HKD) and South Korean Won (KRW).
  - Requires multilingual NLP capabilities (Cantonese/Traditional Chinese, Korean, English).
  - Demonstrates cross-market generalizability to the CrowdWisdomTrading evaluation team.

---

## ADR-002: Selection of 5 Liquid Assets per Exchange
- **Status**: Proposed
- **Context**: The assignment requires scraping social media posts about 5 different assets per exchange covering the last 10 days.
- **Decision**: Select liquid, high-retail-interest mega-caps:
  - **HKEX**: `0700.HK` (Tencent), `9988.HK` (Alibaba), `3690.HK` (Meituan), `1810.HK` (Xiaomi), `1211.HK` (BYD).
  - **KRX**: `005930.KS` (Samsung Electronics), `000660.KS` (SK Hynix), `035420.KS` (NAVER), `035720.KS` (Kakao), `005380.KS` (Hyundai Motor).
- **Consequences**:
  - Guarantees high social chatter volume on LIHKG, Naver, and Telegram.
  - Ensures clean market price bars without illiquidity gaps or zero-volume trading days.
  - Generates robust Exa financial news results.

---

## ADR-003: Selection of Market Price Provider (`yfinance` + `FinanceDataReader`)
- **Status**: Decided
- **Context**: The assignment requires aligning social posts and news with market price bars, but does not prescribe a price data provider.
- **Decision**: Use `yfinance` as the primary market data connector, complemented by `FinanceDataReader` as a specialized Korean fallback connector.
- **Alternatives Considered**:
  - *Alpha Vantage*: Rejected due to strict 25 requests/day free tier limit.
  - *Polygon.io*: Rejected due to lack of standard free-tier KRX/HKEX coverage.
  - *EODHD*: Rejected due to trial expiration risks during evaluation.
- **Consequences**: Zero API key requirement for market prices, high reliability for daily/intraday bars, and total reproducibility for the evaluator.

---

## ADR-004: Social Media Sources & Apify Actors Selection
- **Status**: Decided
- **Context**: The assignment requires at least two independent social sources per exchange using Apify.
- **Decision**:
  - **Hong Kong (HKEX)**:
    1. *LIHKG*: Hong Kong’s dominant retail forum (via Apify LIHKG scraper / Cheerio scraper).
    2. *Telegram*: Public Hong Kong stock channels (via Apify `curious_coder/telegram-scraper` or public channel scraper).
    *Fallback*: AAStocks financial community boards.
  - **South Korea (KRX)**:
    1. *Naver Finance 종목토론실*: Epicenter of KR retail sentiment (via Apify Cheerio web scraper).
    2. *Telegram Korean financial alert channels* or *Daum Finance Forums*.
- **Consequences**: Full compliance with the assignment constraint requiring 2 distinct sources per exchange.

---

## ADR-005: Target Formulation — Return ($R_{t+1}$) vs Raw Price ($P_{t+1}$)
- **Status**: Decided
- **Context**: The assignment states *"predict the price on n+1"*. In financial econometrics, training regression models directly on non-stationary raw price levels ($P_t$) produces spurious correlations, unit root bias, and trivial identity mappings ($\hat{P}_{t+1} \approx P_t$) that mask actual performance.
- **Decision**:
  1. Train models on the stationary percentage return target:
     $$R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$
  2. Compute the implied predicted price:
     $$\hat{P}_{t+1} = P_t \times (1 + \hat{R}_{t+1})$$
- **Consequences**: Statistically sound model learning, meaningful feature weights, and full fulfillment of the user requirement to output predicted price levels.

---

## ADR-006: Database Storage — SQLite with SQLAlchemy ORM
- **Status**: Decided
- **Context**: Need a storage layer that is zero-friction for local developer reproduction, but robust enough to support structured querying, relations, and future PostgreSQL migration.
- **Decision**: Use SQLite (`data/sentiment_agent.db`) via SQLAlchemy 2.0 ORM.
- **Alternatives Considered**:
  - *PostgreSQL*: Excellent for production, but introduces Docker/daemon setup overhead for the reviewer.
  - *Flat Parquet/CSV*: Inefficient for atomic provenance logging and relational joins between posts, sentiment scores, and price bars.
- **Consequences**: Single-file database with instant setup; identical SQL schema ready for PostgreSQL if required.

---

## ADR-007: Multilingual Sentiment Architecture (FinBERT + XLM-RoBERTa)
- **Status**: Decided
- **Context**: Financial news is primarily in English, while retail discussions are in Traditional Chinese (Cantonese slang) and Korean. Standard English-only sentiment models fail on CJK text.
- **Decision**: Implement a two-tier NLP pipeline:
  1. `ProsusAI/finbert` for Exa English news articles.
  2. `cardiffnlp/twitter-xlm-roberta-base-sentiment` (or dedicated KoBERT / Chinese BERT) for social media posts.
- **Consequences**: Accurate domain-specific scoring for news, combined with robust colloquial social media sentiment extraction.

---

## ADR-008: Chronological Walk-Forward Validation
- **Status**: Decided
- **Context**: Standard machine learning practices (e.g. random $K$-fold cross-validation or `train_test_split(shuffle=True)`) cause catastrophic data leakage in time-series forecasting.
- **Decision**: Enforce an expanding-window walk-forward validation strategy with an embargo period. Training sets strictly precede testing sets in chronological time.
- **Consequences**: True out-of-sample realism, zero lookahead leakage, and trustworthy evaluation metrics.

---

## ADR-009: Source-Level Sentiment Isolation Before Blending
- **Status**: Decided
- **Context**: Mixing raw tweets, forum threads, and news into a single bag-of-words obscures source-level dynamics and prevents isolating which channel actually drives retail order flow.
- **Decision**: Compute sentiment scores separately for each source platform (LIHKG, Naver, Telegram, Exa) with individual engagement weights before computing the synthesized composite sentiment score.
- **Consequences**: Enables comparative source attribution in the dashboard and prevents high-volume bot spam on one platform from corrupting institutional news signals.

---

## ADR-010: Visualization Framework — Streamlit
- **Status**: Decided
- **Context**: Need a rapid, interactive, and visually compelling reporting interface to display asset rankings, charts, news feeds, and prediction metrics.
- **Decision**: Use **Streamlit** with Plotly interactive charts.
- **Consequences**: Python-native, fast to build, highly interactive, and easily run with `streamlit run src/dashboard/app.py`.

