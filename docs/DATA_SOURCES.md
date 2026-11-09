# Data Sources, APIs & Asset Coverage Specification

## 1. Target Exchanges & Asset Universe

The assignment specifies coverage of **5 distinct assets per exchange** across two regional hubs with active retail investor sentiment:
1. **Hong Kong Stock Exchange (HKEX / Hang Seng)**
2. **Korea Exchange (KRX / KOSPI)**

### 1.1. Hong Kong Stock Exchange (HKEX) Selection

| # | Ticker | Company Name | Sector | Currency | Selection Criteria & Sentiment Rationale |
|---|---|---|---|---|---|
| 1 | `0700.HK` | Tencent Holdings Ltd | Tech / Digital Entertainment | HKD | Hang Seng bellwether; consistently #1 discussed stock on LIHKG and Telegram HK groups. |
| 2 | `9988.HK` | Alibaba Group Holding | E-Commerce / Cloud | HKD | High-beta retail favorite with massive daily institutional and retail commentary on Exa and LIHKG. |
| 3 | `3690.HK` | Meituan | Local Services / Tech | HKD | High retail volatility, heavy option trading buzz, frequent discussions on Futu/Moomoo and LIHKG. |
| 4 | `1810.HK` | Xiaomi Corporation | Consumer Tech / EV | HKD | Enormous retail following across Asia; intense engagement around quarterly hardware/EV delivery numbers. |
| 5 | `1211.HK` | BYD Company Ltd | Automotive / EV / Clean Tech | HKD | Leading global EV manufacturer; high cross-border retail focus and frequent global news coverage on Exa. |

### 1.2. Korea Exchange (KRX / KOSPI) Selection

| # | Ticker | Company Name | Sector | Currency | Selection Criteria & Sentiment Rationale |
|---|---|---|---|---|---|
| 1 | `005930.KS` | Samsung Electronics Co. | Semiconductors / Tech | KRW | The undisputed retail anchor of the Korean market; #1 most active board on Naver 종목토론실. |
| 2 | `000660.KS` | SK Hynix Inc. | AI Memory Semiconductors | KRW | Massive retail and institutional momentum driven by global HBM/AI hardware demand; active on Blind/Telegram. |
| 3 | `035420.KS` | NAVER Corporation | Search / AI / Fintech | KRW | Native host of Naver Finance; huge organic user discussions on its own corporate performance. |
| 4 | `035720.KS` | Kakao Corporation | Messaging / Internet / Fintech | KRW | Dominant messaging ecosystem; retail-heavy shareholder base with vocal sentiment on Daum Finance and KakaoTalk. |
| 5 | `005380.KS` | Hyundai Motor Company | Automotive / Mobility | KRW | Top chaebol industrial titan; high retail engagement around EV transition and shareholder return policies. |

---

## 2. Social Media Sources & Apify Actors

Per assignment instructions, at least **two independent social sources per exchange** are required:

### 2.1. Hong Kong Exchange (HKEX) Social Sources

#### Source 1: LIHKG (Lihkg.com)
- **Description**: The primary local Cantonese discussion forum in Hong Kong, featuring a dedicated Channel 17 (Finance & Investment). Retail traders actively debate daily swings and rumors.
- **Apify Actor Approach**:
  - Community Actor: `claude_code_reviewer/lihkg-scraper` or dedicated Actor configuration using `apify/web-scraper` / `apify/cheerio-scraper`.
  - Target URL: `https://lihkg.com/category/17` and search endpoints for target tickers (`0700`, `9988`, `3690`, `1810`, `1211`).
  - Extracted Fields: Thread Title, Post Content, Post ID, Timestamp (UTC converted), Upvotes (`like_count`), Downvotes (`dislike_count`), Reply Count, Author ID.

#### Source 2: Telegram HK Financial Channels & Groups
- **Description**: High-velocity, real-time peer discussion and market alert channels operated by Hong Kong retail communities.
- **Apify Actor Approach**:
  - Community Actor: `curious_coder/telegram-scraper` or public channel preview scraper (`apify/telegram-scraper` / `telegram-messages-scraper`).
  - Target Channels: Public Hong Kong stock chat rooms and signal groups (e.g. `@hkstockdiscussions`, public financial broadcast feeds).
  - Extracted Fields: Message Text, Message ID, Channel Name, Timestamp (UTC), View Count, Forward Count, Reactions (emoji counts), Media presence.
- **Alternative / Complementary Source**: **AAStocks Financial Portal Discussions** (`apify/cheerio-scraper` targeting `aastocks.com` stock comment pages).

---

### 2.2. Korea Exchange (KRX) Social Sources

#### Source 1: Naver Finance Stock Discussion Boards (종목토론실)
- **Description**: The epicenter of Korean retail investor sentiment. Every KRX stock has a dedicated, high-traffic discussion board under `https://finance.naver.com/item/board.naver?code={ticker}`.
- **Apify Actor Approach**:
  - Actor: `apify/cheerio-scraper` or custom Apify web crawler.
  - Ingestion Configuration:
    - Base URL: `https://finance.naver.com/item/board.naver?code={code}` where `{code}` is e.g. `005930`.
    - Page Extraction: Post Title, Author Handle, Timestamp (`YYYY.MM.DD HH:MM`), View Count (`조회`), Upvotes (`공감`), Downvotes (`비공감`), Full Text Body.
  - Extracted Fields: Normalized to UTC timestamp, post ID, text, upvotes, downvotes, views.

#### Source 2: Telegram Korean Stock Channels & Trading Alerts
- **Description**: Korean retail and quant investor channels providing real-time peer reactions, breaking news alerts, and sentiment discussions.
- **Apify Actor Approach**:
  - Actor: `curious_coder/telegram-scraper` targeting public Korean financial channels.
  - Extracted Fields: Post ID, Text (Korean), Timestamp (UTC), Views, Forwards, Reactions.
- **Alternative / Complementary Source**: **Daum Finance Forums** (`finance.daum.net` Agora discussion boards).

---

## 3. Financial News Ingestion: Exa.ai

The assignment mandates **Exa.com** for news sentiment.

### 3.1. Query Strategy & Search Pipeline
Exa provides neural semantic search optimized for retrieving dense, relevant web knowledge.
- **API Client**: `exa-py` (`Exa(api_key=settings.EXA_API_KEY)`)
- **Query Formulation**:
  - For HKEX: `"{Ticker} {Company_Name} earnings stock price Hong Kong market"`
    - e.g., `"0700 Tencent Holdings earnings stock price Hong Kong market"`
    - e.g., `"9988 Alibaba cloud revenue e-commerce stock"`
  - For KRX: `"{Ticker} {Company_Name} Korea stock market semiconductor earnings"`
    - e.g., `"005930 Samsung Electronics semiconductor revenue earnings"`
    - e.g., `"000660 SK Hynix HBM AI memory stock rally"`
- **Parameters**:
  - `start_published_date`: Today minus 10 days (UTC ISO string)
  - `end_published_date`: Current UTC timestamp
  - `num_results`: 10 to 25 articles per asset
  - `use_autoprompt`: True
  - `highlights`: True (extracts key snippets matching query context)
  - `text`: True (full article body for accurate NLP analysis)
- **Extracted Fields**:
  - `id`: Exa unique ID
  - `title`: Article title
  - `url`: Canonical web link
  - `published_date`: ISO 8601 published date
  - `text`: Clean article body
  - `score`: Exa neural relevance score

---

## 4. Market Price Provider: Selection & Analysis

The assignment requires price bars aligned with social sentiment, but does not prescribe a specific price API.

### 4.1. Comparative Provider Analysis

| Provider | HKEX Coverage | KRX Coverage | Granularity | Free Tier / Cost | Recommendation |
|---|---|---|---|---|---|
| **Yahoo Finance (`yfinance`)** | Full (`0700.HK`, etc.) | Full (`005930.KS`, etc.) | Daily (`1d`), Hourly (`1h`), 15m | Free, no API key required, reliable OHLCV | **Primary Choice** |
| **FinanceDataReader (KRX library)** | Limited | Native KRX / Naver Finance | Daily (`1d`) | Free Python library | **KRX Fallback** |
| **Alpha Vantage** | Supported | Incomplete / Paid | Daily | Rate-limited (25 req/day free) | Rejected |
| **Polygon.io** | Paid add-on | Not available on basic tier | High | Paid subscription required | Rejected |
| **EODHD** | Supported | Supported | Daily / Intraday | Limited free demo | Rejected |

### 4.2. Selected Solution: `yfinance` + `FinanceDataReader` Fallback
- **Primary Engine**: `yfinance`
  - Fetches 10-day lookback + 60-day historical warm-up window (needed for 14-period RSI, 26-period MACD, and 20-period moving averages).
  - Ticker mappings:
    - HK: `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`, `1211.HK`
    - KR: `005930.KS`, `000660.KS`, `035420.KS`, `035720.KS`, `005380.KS`
  - Validates split-adjusted closing prices and calculates volume metrics.
- **Secondary Engine**: `FinanceDataReader` used as a resilient fallback for KRX tickers if Yahoo Finance encounters transient network throttling.

---

## 5. Rate Limits, Credentials & Provenance Tracking

### 5.1. Credentials Required
The system loads all credentials from `.env`:
- `APIFY_API_TOKEN`: Personal API token from `https://console.apify.com/account/integrations`.
- `EXA_API_KEY`: API key from `https://dashboard.exa.ai/api-keys`.
- `DATABASE_URL`: `sqlite:///data/sentiment_agent.db` (or PostgreSQL connection string).

### 5.2. Rate Limits & Budget Optimization
- **Apify**: Free tier includes $5 monthly credit. Apify actor runs will be batched by asset and cached locally in SQLite to prevent re-scraping previously ingested periods.
- **Exa**: Free tier includes $10 credits (approx. 1,000 requests). Each ticker run consumes 1 targeted search call per 10-day lookback.
- **Local Cache**: Once raw data is stored in `social_posts` and `news_articles`, subsequent feature extraction and ML runs pull directly from the local database, enabling offline reproduction without burning API credits.

### 5.3. Reproducibility Guarantee for Submission
To satisfy Gilad's requirement:
> *"Submit APIFY /exa tokens and actor IDs used for scraping (so we can reproduce your pipeline)"*
The repository will include:
1. Exact Apify Actor IDs in `src/config.py` and `docs/DATA_SOURCES.md`.
2. Seed scripts and database export (`data/sentiment_agent.db` snapshot) allowing instant evaluation even if third-party credentials expire.
3. Provenance audit logs in the `pipeline_runs` table.

