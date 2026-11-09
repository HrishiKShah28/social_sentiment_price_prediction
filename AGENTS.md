# AGENTS.md

## Project Mission & Identity
You are working on **HengSeng Sentiment Price Prediction Agent**, a production-grade financial intelligence and quantitative forecasting system built for CrowdWisdomTrading. The system ingests alternative data (social media sentiment from Apify and financial news from Exa), aligns it with historical market price bars, extracts entity signals and technical levels, and trains an out-of-sample validated predictive model for next-period asset price ($n+1$).

---

## Non-Negotiable Engineering Rules

### 1. Data Integrity & Anti-Fabrication
- **NEVER** fabricate scraped social posts, news articles, market prices, or backtest results.
- **NEVER** present mock or synthetically generated numbers as real scraped data.
- When APIs are unavailable or credentials are not yet configured, the system must fail gracefully with explicit error messages and data provenance logs.
- All scraped items must retain raw payloads, source actor IDs, URLs, and source platform timestamps.

### 2. Time-Series Leakage Prevention
- **NO RANDOM TRAIN-TEST SPLITTING**: Financial time-series data must never be shuffled or randomly partitioned with `train_test_split()`.
- Use strictly chronological, walk-forward validation (expanding or rolling window) with purging and embargo.
- Timestamp alignment must enforce strict point-in-time constraints: features available at cutoff time $t$ can only use information timestamped $\le t$ to predict target $t+1$.
- Market close and trading session boundaries must be respected in UTC.

### 3. Security & Credentials
- **NEVER** hardcode API keys, tokens, or credentials in any file committed to version control.
- All secrets (`APIFY_API_TOKEN`, `EXA_API_KEY`, etc.) must be read exclusively from environment variables via `.env` and validated with `pydantic-settings`.
- Provide a clean `.env.example` with detailed instructions.

### 4. Code Standards & Architecture
- **Language**: Python 3.11+.
- **Modularity**: Clean separation of concerns:
  - `src/ingestion/`: Data extraction connectors (Apify, Exa, Market Data).
  - `src/storage/`: Database connection, models, and repositories.
  - `src/nlp/`: Sentiment scoring, language normalization, and signal extraction.
  - `src/features/`: Time alignment, feature engineering, and composite score computation.
  - `src/models/`: Predictive models, baselines, and walk-forward evaluators.
  - `src/dashboard/`: Interactive Streamlit visualization application.
  - `src/pipeline/`: End-to-end orchestration CLI.
- **Typing**: Use static type hints (`typing` / Python 3.10+ native types) for all public functions and class methods.
- **Error Handling**: Use structured logging (`loguru` or Python `logging`) and custom exceptions. Avoid bare `except:` blocks.
- **Testing**: Every major module must have unit tests in `tests/` using `pytest`.

---

## Directory Layout Reference
```
hengseng-sentiment-agent/
├── assignment.pdf                  # Original assignment specification
├── AGENTS.md                       # Coding conventions & instructions for AI agents
├── README.md                       # Project overview & running instructions
├── pyproject.toml                  # Project dependencies & build config
├── .env.example                    # Template for required environment variables
├── docs/                           # Architecture, PRD, ML strategy, and ADRs
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── DATABASE_SCHEMA.md
│   ├── DATA_SOURCES.md
│   ├── ML_STRATEGY.md
│   ├── DEVELOPMENT_PLAN.md
│   └── DECISIONS.md
├── src/                            # Application source code
│   ├── config.py                   # Pydantic settings & environment configuration
│   ├── ingestion/                  # Apify, Exa, and market price connectors
│   ├── storage/                    # Database models, connection, and queries
│   ├── nlp/                        # Sentiment analysis & technical level extractors
│   ├── features/                   # Time-series alignment & feature engineering
│   ├── models/                     # Predictors, baselines, and walk-forward evaluator
│   ├── dashboard/                  # Streamlit visual dashboard
│   └── pipeline/                   # Master orchestrator CLI
├── data/                           # Local SQLite database & raw data cache (git-ignored)
└── tests/                          # Automated unit, integration, and leakage tests
```

---

## Workflow for Future Agent Turns
1. **Always read existing documentation** in `docs/` before making architectural or structural changes.
2. **Consult `docs/DECISIONS.md`** to understand why specific frameworks, models, or data sources were chosen.
3. **Run existing tests** (`pytest`) before and after modifying code.
4. **Log data provenance** for all ingested data to facilitate reproducibility for the evaluation team.

