# Development & Implementation Plan (5–7 Days)

## 1. Timeline & Phasing Overview

The project is structured into **6 disciplined phases** designed to meet the 5–7 day internship submission deadline, ensuring high engineering rigor, zero fabrication, and reproducible verification.

```
Day 1: Phase 1 — Architecture, Design & Documentation [COMPLETED]
Day 2: Phase 2 — Ingestion Engine (Apify, Exa, Market Data & SQLite)
Day 3: Phase 3 — Sentiment Intelligence, Multi-Source Scoring & Technical Levels
Day 4: Phase 4 — Feature Engineering, Walk-Forward ML Engine & Baselines
Day 5: Phase 5 — Interactive Streamlit Dashboard & Ranking Report
Day 6: Phase 6 — Integration Testing, Provenance Validation & Edge-Case Hardening
Day 7: Phase 7 — Final Deliverables, Demo Recording & Submission Package
```

---

## 2. Phase-by-Phase Breakdown

### Phase 1: Architecture & Documentation (Day 1) — *Current Phase*
- [x] Ingest and analyze complete assignment specification from `assignment.pdf`.
- [x] Create project structure: `AGENTS.md`, `README.md`, `docs/`.
- [x] Document Product Requirements (`docs/PRD.md`) with explicit acceptance criteria.
- [x] Document System Architecture & Data Flow (`docs/ARCHITECTURE.md`).
- [x] Document Database Schema (`docs/DATABASE_SCHEMA.md`) with Mermaid ER diagrams.
- [x] Document Data Sources, Apify Actors & Exa Queries (`docs/DATA_SOURCES.md`).
- [x] Document ML Strategy, Sentiment Scoring, and Walk-Forward Validation (`docs/ML_STRATEGY.md`).
- [x] Document Technical Decisions & Trade-Offs (`docs/DECISIONS.md`).
- [x] Seek user sign-off on architecture and open questions.

---

### Phase 2: Data Ingestion Layer (Day 2)
**Goal**: Build robust, rate-limited connectors for Apify, Exa, and market price bars with SQLite storage.
- **Tasks**:
  1. Initialize Python environment: `pyproject.toml`, poetry/uv or pip dependencies, `.env.example`.
  2. Implement `src/config.py` using `pydantic-settings`.
  3. Implement `src/storage/database.py` and `src/storage/models.py` (SQLAlchemy 2.0 ORM tables).
  4. Build `src/ingestion/price_client.py`:
     - Download OHLCV bars for the 10 assets via `yfinance` with split adjustments.
     - Validate data cleanliness (no negative prices, valid chronological order).
  5. Build `src/ingestion/apify_client.py`:
     - Implement Apify API runner for target actors (LIHKG, Naver Finance, Telegram).
     - Store full raw payloads in `social_posts.raw_payload` with UTC timestamp normalization.
  6. Build `src/ingestion/exa_client.py`:
     - Implement Exa financial news search with semantic query filters.
     - Store highlights and full text in `news_articles`.
  7. Implement CLI command `python -m src.pipeline.run --stage ingest`.
- **Milestone 2 Deliverable**: Database populated with real historical price bars, social posts, and news articles for all 10 assets over the 10-day lookback window.

---

### Phase 3: Sentiment Intelligence & Technical Signal Extraction (Day 3)
**Goal**: Implement multi-language sentiment scoring, engagement weighting, and algorithmic support/resistance detection.
- **Tasks**:
  1. Build `src/nlp/sentiment.py`:
     - Load `ProsusAI/finbert` for English financial news.
     - Load `cardiffnlp/twitter-xlm-roberta-base-sentiment` (or dedicated Korean/Chinese pipeline) for social forum posts.
     - Compute $[-1, +1]$ polarity scores and confidence.
  2. Implement engagement weighting formula:
     $$\omega_i = \ln(1 + \text{likes} + 2\text{reposts} + 0.1\text{views}) \times \ln(1 + \text{followers} + 10)$$
  3. Implement separate source-level sentiment aggregation before composite blending.
  4. Build `src/nlp/levels.py`:
     - Compute rolling swing pivot support ($S_1, S_2$) and resistance ($R_1, R_2$).
     - Implement volume-by-price histogram cluster detection.
     - Formulate transparent rule-based direction (`Bullish`, `Bearish`, `Neutral`) and position recommendation (`Long`, `Short`, `Flat`).
  5. Write unit tests for sentiment normalization and level calculation.
- **Milestone 3 Deliverable**: All ingested posts and news scored in `sentiment_scores`; technical levels and positions computed in `technical_levels`.

---

### Phase 4: Feature Engineering & Walk-Forward ML Engine (Day 4)
**Goal**: Build point-in-time feature matrix and train out-of-sample models predicting next-period price $P_{t+1}$.
- **Tasks**:
  1. Build `src/features/aligner.py`:
     - Enforce strict cutoff rules (market close UTC) ensuring zero future data in feature matrix $X_t$.
  2. Build `src/features/builder.py`:
     - Calculate technical indicators: RSI(14), MACD, ATR, Historical Volatility, Volume ratio.
     - Calculate sentiment indicators: Sentiment momentum, buzz velocity, sentiment-price divergence.
     - Define target: $R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$ and $P_{t+1}$.
  3. Build `src/models/baselines.py`:
     - Baseline 1: Naive Random Walk / Persistence ($\hat{P}_{t+1} = P_t$).
     - Baseline 2: Technical-Only Regressor (no sentiment).
     - Baseline 3: Sentiment-Only Regressor (no price features).
  4. Build `src/models/predictors.py`:
     - Ridge Regressor with $L_2$ shrinkage.
     - XGBoost / LightGBM Regressor.
  5. Build `src/models/validation.py`:
     - Chronological walk-forward expanding window with 1-period embargo.
     - Calculate RMSE, MAE, MAPE, and Directional Hit Rate.
- **Milestone 4 Deliverable**: Backtest table comparing baselines against XGBoost/Ridge with verified out-of-sample metrics.

---

### Phase 5: Visual Dashboard & Ranking Report (Day 5)
**Goal**: Deliver an institutional-grade Streamlit application fulfilling all visual reporting requirements.
- **Tasks**:
  1. Build `src/dashboard/app.py`:
     - **Exchange Leaderboard Tab**:
       - Side-by-side comparison of Hang Seng (HKEX) vs KRX.
       - Interactive sortable ranking table by social mentions, sentiment score, direction, and position.
     - **Asset Deep-Dive Tab**:
       - Interactive Plotly candlestick chart with support/resistance horizontal lines.
       - Overlaid sentiment indicator bar chart.
       - Live feed of actual scraped social posts and news articles with provenance links.
       - Predicted price vs actual price trajectory with error bands.
     - **Model Performance Tab**:
       - Benchmark comparison table (Random Walk vs Technical vs Sentiment vs XGBoost).
       - Directional hit-rate visualization.
       - Walk-forward fold evaluation progression.
- **Milestone 5 Deliverable**: Working Streamlit dashboard launched via `streamlit run src/dashboard/app.py`.

---

### Phase 6: Hardening, Provenance & Quality Assurance (Day 6)
**Goal**: End-to-end testing, error recovery verification, and packaging.
- **Tasks**:
  1. Execute full unit and integration test suite (`pytest tests/`).
  2. Run explicit **Data Leakage Test**: verify that shifting target timestamps fails the validation check.
  3. Verify Apify Actor IDs and Exa queries are recorded in SQLite `pipeline_runs`.
  4. Prepare seed database snapshot (`data/seed_sample.db`) so evaluators can run the dashboard immediately even without entering live API tokens.
  5. Format code with `black` and `ruff`.

---

### Phase 7: Final Documentation & Submission (Day 7)
**Goal**: Submission readiness for CrowdWisdomTrading CEO Gilad.
- **Tasks**:
  1. Finalize `README.md` with:
     - Architecture overview.
     - Quickstart guide (with and without API keys).
     - Apify Actor IDs and Exa configuration guide.
     - Model performance evaluation summary.
  2. Verify `.env.example` includes all required environment keys.
  3. Record a 3-minute video/GIF walkthrough of the Streamlit dashboard and pipeline CLI.
  4. Prepare email draft to `gilad@crowdwisdomtrading.com`.

---

## 3. Risk Matrix & Contingency Planning

| Risk | Impact | Probability | Mitigation Strategy |
|---|---|---|---|
| **Apify rate limits / credit exhaustion** | High | Low | Batch actor runs; cache raw responses in SQLite immediately; include offline seed snapshot. |
| **LIHKG / Naver anti-scraping changes** | Medium | Medium | Use multi-source redundancy (Telegram + AAStocks for HK; Daum + Telegram for KR); modular actor adapters. |
| **Zero news on specific low-profile ticker** | Low | Low | Chosen assets are mega-caps (Tencent, Samsung, Alibaba) with guaranteed daily global and local news flow. |
| **Model overfitting on 10-day lookback** | High | Medium | Predict returns ($R_{t+1}$) instead of raw price; use strong $L_2$ regularization (Ridge) and shallow trees (XGBoost max_depth=3); report out-of-sample metrics only. |

