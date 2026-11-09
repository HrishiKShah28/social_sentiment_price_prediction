# Product Requirements Document (PRD)

## Project: HengSeng Sentiment Price Prediction Agent
**Client / Evaluator**: CrowdWisdomTrading (Gilad, CEO)  
**Role**: AI Agents Developer (Intern Position Assessment)  
**Timeline**: 5–7 Days  
**Status**: Architecture & Design Phase  

---

## 1. Executive Summary & Objective
The goal is to build an institutional-grade, modular Python financial intelligence and quantitative forecasting agent that predicts an asset's next-period price ($n+1$) by synthesizing alternative sentiment data from at least two independent social media sources per exchange, complemented by institutional financial news, and aligned with historical market price bars.

The system will ingest data for **5 distinct assets across 2 target exchanges** (Hong Kong HKEX / Hang Seng and South Korea KRX) over the **most recent 10 days**, extract sentiment, direction, trading position, and support/resistance levels, rank assets visually, and evaluate price predictions using a leakage-free walk-forward time-series validation framework.

---

## 2. Stakeholders & Success Criteria
- **Primary Stakeholder**: Gilad, CEO at CrowdWisdomTrading (`gilad@crowdwisdomtrading.com`).
- **Success Criteria**:
  1. Complete ingestion pipeline utilizing **Apify** (for social platforms) and **Exa** (for financial news).
  2. Integration of at least **two independent social sources per exchange**.
  3. Extraction of granular post metadata: post text, UTC timestamps, likes, reposts, views, and author follower counts.
  4. Point-in-time alignment with historical price bars without lookahead bias.
  5. Per-ticker extraction of: ticker, asset name, current price, composite sentiment, direction (Bullish/Bearish/Neutral), suggested trading position, and calculated support/resistance levels.
  6. Rich interactive visual report (Streamlit) ranking tickers by mentions, sentiment, and direction.
  7. Next-period price prediction ($n+1$) benchmarked against standard baselines via chronological out-of-sample validation.
  8. Full reproducibility with documented API tokens, Apify Actor IDs, and data provenance.

---

## 3. Scope of Work

### In-Scope
- Ingestion of social media data via Apify for 5 HKEX assets and 5 KRX assets covering the last 10 days.
- Ingestion of financial news articles via Exa for the same 10 assets covering the last 10 days.
- Ingestion of historical market OHLCV price bars.
- Storage of raw and processed data in a relational database (SQLite for local reproducibility, PostgreSQL compatible).
- Sentiment scoring using domain-specific/multilingual NLP models handling Traditional Chinese, Korean, and English.
- Composite sentiment aggregation weighting author influence, engagement, and source credibility.
- Technical analysis engine computing support/resistance levels, trend directions, and position signals.
- Feature engineering combining lagged technical indicators and sentiment metrics.
- Machine learning models predicting next-period price return / price with walk-forward validation.
- Interactive Streamlit dashboard with asset rankings, drill-down charts, and backtest evaluation.
- Clean CLI for pipeline orchestration.

### Out-of-Scope (Non-Goals)
- Live automated order execution or broker connectivity (this is a research/intelligence system).
- High-frequency microsecond trading (target granularity is daily or multi-hour bars).
- Commercial production multi-tenant SaaS deployment.
- Bypassing authentication gates or anti-bot mechanisms outside documented Apify/API capabilities.

---

## 4. Functional Requirements

### 4.1. Asset & Exchange Coverage (FR-1)
- **FR-1.1**: The system must cover two distinct exchanges:
  - **Exchange 1**: Hong Kong Stock Exchange (HKEX / Hang Seng).
  - **Exchange 2**: Korea Exchange (KRX / KOSPI).
- **FR-1.2**: Exactly 5 representative, liquid assets must be tracked per exchange:
  - **HKEX Assets**:
    1. `0700.HK` (Tencent Holdings Ltd)
    2. `9988.HK` (Alibaba Group Holding Ltd)
    3. `3690.HK` (Meituan)
    4. `1810.HK` (Xiaomi Corporation)
    5. `1211.HK` (BYD Company Ltd)
  - **KRX Assets**:
    1. `005930.KS` (Samsung Electronics Co., Ltd.)
    2. `000660.KS` (SK Hynix Inc.)
    3. `035420.KS` (NAVER Corporation)
    4. `035720.KS` (Kakao Corporation)
    5. `005380.KS` (Hyundai Motor Company)

### 4.2. Alternative Data Ingestion (FR-2)
- **FR-2.1**: **Apify Social Ingestion**:
  - Must use Apify Actors to collect posts for each asset over the past 10 days.
  - Must use at least 2 independent social sources per exchange:
    - **HKEX Sources**: LIHKG (Hong Kong retail forum) + Telegram HK financial channels (or AAStocks forum).
    - **KRX Sources**: Naver Finance Stock Discussion Boards (종목토론실) + Telegram Korean equity channels (or Daum Finance).
  - Must extract: `post_id`, `text`, `timestamp_utc`, `likes`, `reposts`, `views`, `author_follower_count`, `author_id`, `url`.
- **FR-2.2**: **Exa Financial News Ingestion**:
  - Must use Exa API (`exa-py`) to ingest high-relevance financial news for each asset over the past 10 days.
  - Must extract: `article_id`, `title`, `url`, `snippet`/`text`, `published_date_utc`, `domain`, `exa_relevance_score`.
- **FR-2.3**: **Market Price Data Ingestion**:
  - Ingest OHLCV bars (open, high, low, close, volume) for all 10 assets for the required modeling window (including historical warm-up for technical indicators).

### 4.3. Data Preprocessing & Storage (FR-3)
- **FR-3.1**: Normalize all timestamps to ISO 8601 UTC.
- **FR-3.2**: Clean text (strip URLs, HTML tags, normalize Chinese/Korean unicode, preserve financial cashtags and numbers).
- **FR-3.3**: Persist raw payloads and normalized entities in SQLite via SQLAlchemy ORM with proper primary keys, foreign keys, and indexes.

### 4.4. Sentiment Intelligence & Entity Signal Extraction (FR-4)
- **FR-4.1**: Compute sentiment scores on text:
  - English: FinBERT (`ProsusAI/finbert` or specialized financial sentiment pipeline).
  - Traditional Chinese & Korean: Multilingual financial sentiment model (`cardiffnlp/twitter-xlm-roberta-base-sentiment` or language-specific RoBERTa).
- **FR-4.2**: Score output must be normalized to $[-1.0, +1.0]$:
  - Positive: $(0.15, 1.0]$
  - Neutral: $[-0.15, 0.15]$
  - Negative: $[-1.0, -0.15)$
- **FR-4.3**: Compute **Engagement-Weighted Sentiment** per post:
  $$\text{Weight} = \ln(1 + \text{likes} + 2 \times \text{reposts} + 0.1 \times \text{views}) \times \ln(1 + \text{followers})$$
- **FR-4.4**: For each ticker, extract and maintain:
  - Ticker & Asset Name
  - Current Market Price
  - Sentiment Score & Buzz Volume
  - Overall Direction (`Bullish`, `Bearish`, `Neutral`)
  - Trading Position (`Long`, `Short`, `Cash/Flat`)
  - Technical Levels: Support Level ($S_1, S_2$) and Resistance Level ($R_1, R_2$) derived from rolling swing pivots and volume-by-price.

### 4.5. Feature Engineering & Time-Series Alignment (FR-5)
- **FR-5.1**: Point-in-time aggregation: Map all posts and news published during $(t-1, t]$ into bar $t$.
- **FR-5.2**: Technical Indicators: Calculate RSI(14), MACD(12, 26, 9), ATR(14), Historical Volatility(10), 5-day / 20-day Simple Moving Average ratio, Volume change.
- **FR-5.3**: Composite Sentiment Features: Social mention velocity, weighted sentiment mean, sentiment momentum ($\Delta S = S_t - S_{t-1}$), news sentiment mean, sentiment-price divergence.
- **FR-5.4**: Target Variable Formulation:
  - Primary target: Next-period return $R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$
  - Implied predicted price: $\hat{P}_{t+1} = P_t \times (1 + \hat{R}_{t+1})$

### 4.6. Predictive Modeling & Walk-Forward Validation (FR-6)
- **FR-6.1**: Prevent look-ahead bias with strict chronological walk-forward validation (expanding window) with a 1-period embargo.
- **FR-6.2**: Mandatory Baselines:
  - Baseline 1: Naive Random Walk / Persistence ($\hat{P}_{t+1} = P_t$).
  - Baseline 2: Technical-Only Model (features without sentiment).
  - Baseline 3: Sentiment-Only Model (features without technical indicators).
- **FR-6.3**: Main ML Models:
  - Ridge / Lasso Regularized Regression
  - Gradient Boosted Trees (XGBoost / LightGBM)
  - Random Forest Regressor
- **FR-6.4**: Evaluation Metrics:
  - Regression: RMSE, MAE, MAPE, $R^2$.
  - Directional: Directional Accuracy (Hit Rate %): $\frac{1}{N}\sum \mathbb{I}(\text{sign}(\hat{R}_{t+1}) == \text{sign}(R_{t+1}))$.

### 4.7. Visual Reporting & Interactive Dashboard (FR-7)
- **FR-7.1**: Implement a Streamlit dashboard.
- **FR-7.2**: Cross-Exchange Overview: Side-by-side comparison of Hang Seng (HKEX) vs KRX.
- **FR-7.3**: Asset Ranking Table: Sortable by social mentions, sentiment score, direction, and position.
- **FR-7.4**: Single-Asset Drill-Down:
  - Candlestick price chart with support and resistance overlays.
  - Overlaid sentiment indicator bar chart.
  - Interactive news and social feed showing real raw posts, authors, and engagement metrics.
  - Out-of-sample prediction vs actual price plot with confidence interval.
- **FR-7.5**: Model Performance & Validation Tab:
  - Detailed metrics table comparing baselines vs XGBoost/Ridge.
  - Directional hit-rate chart.

---

## 5. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| **NFR-1** | **Reproducibility** | The entire pipeline must be runnable via a single CLI command with deterministic seeds. |
| **NFR-2** | **Security** | Zero API keys committed to Git. All secrets loaded via environment variables. |
| **NFR-3** | **Data Integrity** | Strict provenance tracking (source URL, Actor ID, raw payload hash, ingestion timestamp). |
| **NFR-4** | **Performance** | Pipeline feature generation and model training under 5 minutes for the 10-day lookback window. |
| **NFR-5** | **Modularity** | Decoupled layers (ingestion, storage, NLP, features, models, UI) allowing unit testability. |

---

## 6. Implementation Checklist & Verification Matrix

| Area | Item | Status | Verification Method |
|---|---|---|---|
| **Setup** | Folder structure, `AGENTS.md`, `README.md`, `pyproject.toml` | Pending | File inspection & directory check |
| **Config** | Pydantic Settings reading `.env` for Apify, Exa, DB | Pending | Unit test `test_config.py` |
| **Database** | SQLite + SQLAlchemy schema with 8 normalized tables | Pending | DB migration & ORM schema tests |
| **Ingestion** | Apify Connector with rate-limiting & error recovery | Pending | Apify API integration test with actor IDs |
| **Ingestion** | Exa Connector with date & relevance filtering | Pending | Exa API test with test query |
| **Ingestion** | Market Data Connector (`yfinance`) with bar validation | Pending | Price bar completeness test |
| **NLP** | Multilingual financial sentiment scoring pipeline | Pending | Sentiment test on sample text (ZH/KO/EN) |
| **Analytics** | Support/Resistance level calculator & position logic | Pending | Math verification against price series |
| **Features** | Point-in-time alignment & feature matrix generator | Pending | Leakage test (ensure $t+1$ data never present at $t$) |
| **Modeling** | Baselines + XGBoost / Ridge regressors | Pending | Model fit & predict test |
| **Validation** | Walk-forward cross-validation engine | Pending | Split timestamp monotonicity test |
| **UI** | Streamlit multi-tab dashboard with ranking report | Pending | Streamlit smoke run & UI visual check |
| **Docs** | PRD, ARCHITECTURE, DATABASE, ML, DATA_SOURCES | In Progress | Markdown lint & completeness review |

