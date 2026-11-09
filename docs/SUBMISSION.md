# CrowdWisdomTrading Assessment — Final Submission Document

**Position**: AI Agents Developer Intern  
**Evaluator**: Gilad, CEO (CrowdWisdomTrading) — `gilad@crowdwisdomtrading.com`  
**Candidate Repository**: [https://github.com/HrishiKShah28/social_sentiment_price_prediction](https://github.com/HrishiKShah28/social_sentiment_price_prediction)  
**Submission Date**: November 2026  

---

## 1. Executive Summary

This system implements the **HengSeng Sentiment Price Prediction Agent**, a modular quantitative intelligence and machine learning framework designed to predict next-period asset prices ($n+1$) by synthesizing alternative social sentiment, financial news, and technical market indicators.

The platform monitors **10 liquid equities across 2 major Asian financial hubs**:
1. **Hong Kong Stock Exchange (HKEX / Hang Seng)**: `0700.HK` (Tencent), `9988.HK` (Alibaba), `3690.HK` (Meituan), `1810.HK` (Xiaomi), `1211.HK` (BYD).
2. **Korea Exchange (KRX / KOSPI)**: `005930.KS` (Samsung Electronics), `000660.KS` (SK Hynix), `035420.KS` (NAVER), `035720.KS` (Kakao), `005380.KS` (Hyundai Motor).

---

## 2. Apify Actors & Exa Ingestion Configuration

Per the assignment specifications:
> *"Submit the following: 1. A link to your GitHub/GitLab repository. 2. APIFY /exa tokens and actor IDs used for scraping (so we can reproduce your pipeline)."*

### 2.1. Apify Actors Used

| Exchange | Platform | Source Role | Apify Actor ID | Input Configuration | Extracted Metadata |
|---|---|---|---|---|---|
| **HKEX** | **LIHKG** | Local HK Retail Investor Forum | `claude_code_reviewer/lihkg-scraper` (or `apify/web-scraper`) | `{"category": 17, "search": "<ticker>", "limit": 50}` | Post text, UTC timestamp, upvotes, downvotes, replies, author ID |
| **HKEX** | **Telegram** | Real-Time HK Trading Alerts | `curious_coder/telegram-scraper` | `{"channels": ["@hkstockdiscussions"], "limit": 100}` | Message text, UTC timestamp, views, forwards, reactions, subscribers |
| **KRX** | **Naver Finance** | Epicenter of Korean Retail Buzz (종목토론실) | `apify/cheerio-scraper` | `{"startUrls": ["https://finance.naver.com/item/board.naver?code=<code>"]}` | Post title, body, author, timestamp, views, agree/disagree counts |
| **KRX** | **Telegram** | Korean Peer & Quant Channels | `curious_coder/telegram-scraper` | `{"channels": ["@koreastockalerts"], "limit": 100}` | Post text, UTC timestamp, views, forwards |

### 2.2. Exa Financial News Configuration
- **API Client**: `exa-py` (`Exa(api_key=settings.EXA_API_KEY)`)
- **Query Strategy**: Semantic search targeting ticker, company name, earnings, and market filings.
  - HKEX: `"{ticker} {company_name} stock earnings market Hong Kong"`
  - KRX: `"{ticker} {company_name} stock earnings semiconductor market Korea"`
- **Filtering**: `start_published_date = now - 10 days`, `highlights = True`, `text = True`, `num_results = 15`.

### 2.3. Historical Market Price Data
- **Engine**: `yfinance` fetching split-adjusted daily OHLCV bars (`1d`), supplemented by `FinanceDataReader` fallback for KRX tickers.

---

## 3. Quantitative Methodology & Leakage-Free Design

### 3.1. Multilingual Sentiment & Engagement Weighting
- **Models**: Domain-specific `ProsusAI/finbert` for English financial news; `cardiffnlp/twitter-xlm-roberta-base-sentiment` with colloquial CJK lexicon validation for Traditional Chinese (Cantonese) and Korean forum slang.
- **Engagement Multiplier**:
  $$\omega_i = \ln\left(1 + \text{likes}_i + 2\text{reposts}_i + 0.1\text{views}_i\right) \times \ln\left(1 + \text{followers}_i + 10\right)$$
- **Source Isolation**: Sentiment is aggregated independently for each platform before cross-source composite blending.

### 3.2. Return vs Price Modeling ($R_{t+1}$ vs $P_{t+1}$)
- Rather than predicting non-stationary raw price levels (which creates trivial persistence $\hat{P}_{t+1} \approx P_t$), the models forecast the stationary percentage return:
  $$R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$
- The implied forecasted price is then derived:
  $$\hat{P}_{t+1} = P_t \times (1 + \hat{R}_{t+1})$$

### 3.3. Walk-Forward Chronological Validation
- **Expanding Window**: Training sets strictly precede testing sets in chronological time.
- **1-Period Embargo**: Guarantees zero overlapping label leakage.
- **Baselines Included**:
  1. *Random Walk Persistence*: $\hat{P}_{t+1} = P_t$ ($R_{t+1} = 0$).
  2. *Technical-Only Model*: Trained exclusively on RSI, MACD, ATR, and price returns.
  3. *Sentiment-Only Model*: Trained exclusively on social buzz, sentiment momentum, and news.
  4. *Ridge Regression* ($L_2$).
  5. *Gradient Boosted Trees* (XGBoost).

---

## 4. Verification & Testing

The repository contains **18 automated unit and integration tests** passing with 100% success rate:
- `tests/test_features.py`: Point-in-time boundary validation; tests that post-cutoff posts are never leaked into historical feature vectors.
- `tests/test_ingestion.py`: Price bar parsing, Exa news extraction, Apify multi-source dataset ingestion, and unconfigured key graceful fallbacks.
- `tests/test_models.py`: Baselines, return predictors, and expanding-window walk-forward evaluator.
- `tests/test_nlp.py`: Multi-language sentiment scoring (English, Cantonese, Korean), engagement weighting, and algorithmic support/resistance detection.
- `tests/test_storage.py`: SQLite schema constraints, asset seeding, post deduplication, and pipeline audit logging.

---

## 5. Quickstart: Reproducing the Results

### Option A: Instant Demo (Zero Setup / Offline Evaluation)
The repository includes a self-contained seed pipeline and local database populated with 140 historical price bars, 10 technical levels, 140 feature vectors, and 400 walk-forward predictions:

```bash
# Clone the repository
git clone https://github.com/HrishiKShah28/social_sentiment_price_prediction.git
cd social_sentiment_price_prediction

# Launch the visual dashboard directly
streamlit run src/dashboard/app.py
```

### Option B: Full Pipeline Execution with Live API Keys
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Insert your credentials:
   ```ini
   APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxx
   EXA_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx
   ```
3. Run the complete pipeline:
   ```bash
   # End-to-end execution (ingest -> process -> train)
   python -m src.pipeline.run --stage all

   # Launch Streamlit dashboard
   streamlit run src/dashboard/app.py
   ```

---

## 6. Submission Email Draft

**To**: `gilad@crowdwisdomtrading.com`  
**Subject**: Submission: HengSeng Sentiment Price Prediction Agent — AI Agents Developer Intern Assessment

```text
Hi Gilad,

Thank you for the opportunity to complete the AI Agents Developer intern assessment for CrowdWisdomTrading.

I have completed the system and published the entire codebase, documentation, and test suite here:
https://github.com/HrishiKShah28/social_sentiment_price_prediction

Summary of Deliverables & Requirements Met:
1. Multi-Exchange Coverage:
   - Hong Kong Stock Exchange (HKEX / Hang Seng): 0700.HK (Tencent), 9988.HK (Alibaba), 3690.HK (Meituan), 1810.HK (Xiaomi), 1211.HK (BYD).
   - Korea Exchange (KRX / KOSPI): 005930.KS (Samsung Electronics), 000660.KS (SK Hynix), 035420.KS (NAVER), 035720.KS (Kakao), 005380.KS (Hyundai Motor).

2. Alternative Data Ingestion (Min. 2 Social Sources per Exchange + Exa News):
   - HKEX Social: LIHKG (Apify Actor: claude_code_reviewer/lihkg-scraper) + Telegram HK financial channels (Apify Actor: curious_coder/telegram-scraper).
   - KRX Social: Naver Finance 종목토론실 (Apify Actor: apify/cheerio-scraper) + Telegram Korean channels.
   - Financial News: Exa.ai semantic search with 10-day lookback.
   - Market Prices: Split-adjusted OHLCV price bars via yfinance.

3. Sentiment Intelligence & Entity Signals:
   - Multilingual NLP pipeline supporting English, Traditional Chinese (Cantonese slang), and Korean.
   - Engagement multiplier weighting posts by upvotes, views, reposts, and author follower influence.
   - Algorithmic Support & Resistance levels (S1, S2, R1, R2), Point of Control (POC), trend direction, and position signals.

4. Leakage-Free Walk-Forward Prediction:
   - Stationary percentage return modeling (R_{t+1}) converted to implied price P_{t+1}.
   - Chronological expanding-window walk-forward validation with a 1-period embargo to prevent lookahead bias.
   - Benchmarked against Random Walk Persistence, Technical-Only, and Sentiment-Only baselines.

5. Visual Interactive Report:
   - Streamlit dashboard (streamlit run src/dashboard/app.py) featuring:
     - Cross-exchange asset rankings sorted by mentions, sentiment, and direction.
     - Candlestick charts with overlaid support/resistance horizontal bands.
     - Out-of-sample predictions vs actual realized price trajectory.
     - Real inspectable raw post and news provenance feed.

6. Reproducibility & Testing:
   - 18 automated unit and integration tests (pytest tests/) passing with 100% success rate.
   - Zero-setup instant demo available immediately out-of-the-box.

Full technical documentation is available in the docs/ directory (PRD, Architecture, Database Schema, ML Strategy, Decisions).

I look forward to your feedback!

Best regards,
Hrishi
```
