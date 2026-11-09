# Machine Learning & Quantitative Sentiment Strategy

## 1. Executive Summary
This document defines the quantitative methodology, sentiment engineering, feature construction, target formulation, and leakage-free validation framework for the **HengSeng Sentiment Price Prediction Agent**. 

The core thesis is that **social sentiment captures retail order flow imbalances and behavioral momentum**, while **news sentiment captures fundamental corporate developments**. When properly isolated by source, engagement-weighted, and combined with classical price-action technicals, these alternative signals provide predictive power for next-period price returns ($R_{t+1}$) over standard random-walk baselines.

---

## 2. Sentiment Intelligence & NLP Pipeline

### 2.1. Multi-Language Challenge & Model Selection
Our universe spans three distinct linguistic ecosystems:
1. **English**: Global financial news retrieved via Exa.
2. **Traditional Chinese (Cantonese/Hong Kong)**: Retail discussions on LIHKG and AAStocks.
3. **Korean**: Retail discussions on Naver Finance 종목토론실, Daum, and Telegram.

#### NLP Architecture:
- **English News Sentiment**:
  - Model: `ProsusAI/finbert` (Hugging Face)
  - Pretrained on Financial PhraseBank; outputs 3-class softmax probabilities: $[p_{pos}, p_{neu}, p_{neg}]$.
- **Traditional Chinese & Korean Social Sentiment**:
  - Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` or `klue/roberta-base` (Korean) & `bert-base-chinese`.
  - XLM-RoBERTa is fine-tuned across millions of multilingual social posts, maintaining robust sentiment representations for colloquial forum slang, emojis, and informal syntax.
- **Normalized Polarity Score ($s_i$)**:
  $$s_i = p_{pos} - p_{neg} \in [-1.0, +1.0]$$
  with confidence $c_i = \max(p_{pos}, p_{neu}, p_{neg}) \in [0.0, 1.0]$.
  - Classification thresholds:
    - $\text{Bullish (Positive)}$ if $s_i > +0.15$
    - $\text{Neutral}$ if $-0.15 \le s_i \le +0.15$
    - $\text{Bearish (Negative)}$ if $s_i < -0.15$

---

### 2.2. Engagement-Weighted Social Signal
Not all social posts carry equal information. A post seen by 50,000 traders or authored by an influencer with 100,000 followers conveys vastly more market-moving sentiment than a single bot tweet.

For post $i$:
$$\omega_i = \ln\left(1 + \text{likes}_i + 2 \times \text{reposts}_i + 0.1 \times \text{views}_i\right) \times \ln\left(1 + \text{followers}_i + 10\right)$$

The **Engagement-Weighted Sentiment** for platform $k$ over time bucket $t$ is:
$$\bar{S}_{k, t} = \frac{\sum_{i \in \mathcal{P}_{k, t}} \omega_i \cdot s_i}{\sum_{i \in \mathcal{P}_{k, t}} \omega_i + \epsilon}$$
where $\epsilon = 10^{-6}$ prevents division by zero when post volume is zero.

---

### 2.3. Independent Source Isolation & Composite Sentiment Formula
*Per the user's explicit design rule: separate source-level sentiment must be computed before aggregation.*

For asset $a$ in time interval $(t-1, t]$:
1. **Source 1 Social Sentiment**: $\bar{S}_{\text{source1}, t}$ (e.g. LIHKG or Naver)
2. **Source 2 Social Sentiment**: $\bar{S}_{\text{source2}, t}$ (e.g. Telegram or AAStocks)
3. **News Sentiment**: $\bar{S}_{\text{news}, t}$ (Exa financial articles)

#### Multi-Source Aggregation:
First, combine social sources by activity volume:
$$S_{\text{social}, t} = \frac{N_{\text{src1}, t} \bar{S}_{\text{src1}, t} + N_{\text{src2}, t} \bar{S}_{\text{src2}, t}}{N_{\text{src1}, t} + N_{\text{src2}, t} + \epsilon}$$

Second, synthesize social sentiment with institutional news:
$$S_{\text{composite}, t} = \lambda \cdot S_{\text{social}, t} + (1 - \lambda) \cdot \bar{S}_{\text{news}, t}$$
where $\lambda \in [0.5, 0.7]$ reflects the higher frequency of social sentiment in short-term retail momentum. When news volume is zero in a specific bar, $\lambda \to 1.0$.

---

## 3. Entity Signal Extraction & Technical Analysis

For each ticker, the system programmatically derives the required metadata:

### 3.1. Algorithmic Support & Resistance Levels
To avoid subjective discretionary drawing, support and resistance are computed using **Rolling Swing Pivots** combined with **Volume-Weighted Price Clusters**:
- **Swing Highs / Resistance ($R_1, R_2$)**:
  $$R_1 = \max_{k \in [t-N, t]} High_k, \quad R_2 = \text{Pivot} + (High - Low)$$
  where $\text{Pivot} = \frac{High + Low + Close}{3}$.
- **Swing Lows / Support ($S_1, S_2$)**:
  $$S_1 = \min_{k \in [t-N, t]} Low_k, \quad S_2 = \text{Pivot} - (High - Low)$$
- **Volume Profile Mode**: Identifies horizontal price bins with peak traded volume over the last 10 trading sessions.

### 3.2. Direction & Position Classifier
The system synthesizes technical location and sentiment polarity into a transparent signal:
- **Direction**:
  - `Bullish`: $S_{\text{composite}} > +0.15$ AND $Price_t > \text{SMA}_{20}(Price)$
  - `Bearish`: $S_{\text{composite}} < -0.15$ AND $Price_t < \text{SMA}_{20}(Price)$
  - `Neutral`: Conflicting or muted signals.
- **Position Recommendation**:
  - `Long`: Direction is Bullish AND $Price_t$ is within 2% of Support $S_1$ (favorable risk-reward).
  - `Short`: Direction is Bearish AND $Price_t$ is within 2% of Resistance $R_1$.
  - `Cash / Flat`: Neutral sentiment, or price in middle of range with RSI between 45 and 55.

---

## 4. Time-Series Alignment & Leakage Prevention

### 4.1. Point-in-Time Cutoff & Lookahead Bias Elimination
Financial models frequently fail in production due to subtle lookahead leakage (e.g. using news published at 17:00 UTC to predict a bar that closed at 16:00 UTC).

#### Session Boundary Rules:
- **HKEX Trading Session**: 09:30 to 16:00 HKT (01:30 to 08:00 UTC).
- **KRX Trading Session**: 09:00 to 15:30 KST (00:00 to 06:30 UTC).
- **Alignment Rule**:
  - For predicting bar $t+1$ (Close at Day $t+1$):
  - The feature vector $X_t$ is constructed strictly using:
    - Market price bars up to and including session close of Day $t$.
    - Social posts with `timestamp_utc` $\le \text{CloseTime}(t)$.
    - News articles with `published_date_utc` $\le \text{CloseTime}(t)$.
  - Any post or news item timestamped after $\text{CloseTime}(t)$ is sequestered and assigned to feature vector $X_{t+1}$.

```
Day t Session             Cutoff t             Day t+1 Session             Target t+1
[---- Price Bar t ----]       |             [---- Price Bar t+1 ----]         |
[ Social / News (t-1, t] ]    |                                               |
==============================X===============================================>
     Features X_t             |                                           Target Y_{t+1}
```

---

## 5. Feature Engineering Matrix

For each asset at time $t$, feature vector $X_t \in \mathbb{R}^D$ is assembled:

| Feature Category | Feature Name | Formula / Description |
|---|---|---|
| **Sentiment Level** | `social_sentiment_mean` | Unweighted average polarity of social posts in $(t-1, t]$ |
| | `social_sentiment_weighted` | Engagement-weighted polarity $\bar{S}_{\text{social}, t}$ |
| | `news_sentiment_mean` | Average polarity of Exa news in $(t-1, t]$ |
| | `composite_sentiment` | Combined social + news sentiment score |
| **Sentiment Dynamics** | `sentiment_momentum` | $S_{\text{composite}, t} - S_{\text{composite}, t-1}$ |
| | `social_buzz_velocity` | $\frac{N_{\text{posts}, t}}{\text{MA}_5(N_{\text{posts}})}$ (surge in retail chatter) |
| | `sentiment_dispersion` | Standard deviation of social sentiment in bar (bull/bear disagreement) |
| **Price Technicals** | `return_lag1` | $\frac{P_t - P_{t-1}}{P_{t-1}}$ |
| | `return_lag5` | $\frac{P_t - P_{t-5}}{P_{t-5}}$ |
| | `rsi_14` | 14-period Relative Strength Index |
| | `macd_hist` | MACD(12, 26) - Signal(9) |
| | `atr_14` | Average True Range normalized by $P_t$ |
| | `hist_vol_10` | 10-period annualized standard deviation of log returns |
| | `volume_ratio` | $\frac{\text{Volume}_t}{\text{MA}_5(\text{Volume})}$ |
| **Interaction** | `sentiment_price_divergence` | $Z(S_{\text{composite}, t}) - Z(\text{return\_lag1})$ (contrarian indicator) |

---

## 6. Target Formulation: Return vs Price

### The Non-Stationarity Problem:
Predicting absolute raw price $P_{t+1}$ directly with regression models is a widely recognized trap in quantitative finance:
- Prices are non-stationary (they possess unit roots and random walk drift).
- A model predicting raw price will simply learn the identity function $\hat{P}_{t+1} \approx P_t$, producing deceptively high $R^2 \approx 0.99$ while having zero directional edge.

### The Institutional Solution:
1. **Primary Target Variable**: Next-period percentage return:
   $$R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$
   $R_{t+1}$ is stationary, bounded, and directly reflects economic gain/loss.
2. **Implied Predicted Price ($\hat{P}_{t+1}$)**:
   $$\hat{P}_{t+1} = P_t \times (1 + \hat{R}_{t+1})$$
   This satisfies the assignment requirement to predict $P_{t+1}$ while preserving statistical rigor.

---

## 7. Leakage-Free Validation Strategy

### 7.1. Chronological Walk-Forward Expanding Window
**Standard $K$-Fold cross-validation or random train-test splits are strictly prohibited.**

Instead, the model is evaluated via **Walk-Forward Validation**:
```
Fold 1:  [ Train: Days 1 .. 5 ] -> [ Embargo: Day 5 ] -> [ Test: Day 6 ]
Fold 2:  [ Train: Days 1 .. 6 ] -> [ Embargo: Day 6 ] -> [ Test: Day 7 ]
Fold 3:  [ Train: Days 1 .. 7 ] -> [ Embargo: Day 7 ] -> [ Test: Day 8 ]
Fold 4:  [ Train: Days 1 .. 8 ] -> [ Embargo: Day 8 ] -> [ Test: Day 9 ]
Fold 5:  [ Train: Days 1 .. 9 ] -> [ Embargo: Day 9 ] -> [ Test: Day 10 ]
```
- **Expanding Window**: Training history grows chronologically.
- **Purging & Embargo**: A 1-period gap ensures no forward-looking label overlap.

---

### 7.2. Benchmark Baselines & ML Models

To rigorously demonstrate the value-add of alternative sentiment data, every ML model is benchmarked against three baselines:

1. **Baseline 1: Naive Random Walk / Persistence**:
   $$\hat{P}_{t+1} = P_t \quad (\hat{R}_{t+1} = 0)$$
2. **Baseline 2: Technical-Only Model**:
   - Model trained solely on price, volume, RSI, MACD, and volatility (zero sentiment features).
3. **Baseline 3: Sentiment-Only Model**:
   - Model trained solely on social and news sentiment features (zero technical indicators).
4. **Primary Model 1: Regularized Ridge Regressor**:
   - Linear baseline with $L_2$ regularization to prevent overfitting on small sample sizes.
5. **Primary Model 2: Gradient Boosted Trees (XGBoost / LightGBM)**:
   - Non-linear tree model capturing feature interactions (e.g. high positive sentiment + low RSI = strong breakout).

---

### 7.3. Performance Evaluation Metrics

#### Regression Accuracy:
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N}\sum (\hat{P}_i - P_i)^2}$
- **Mean Absolute Error (MAE)**: $\frac{1}{N}\sum |\hat{P}_i - P_i|$
- **Mean Absolute Percentage Error (MAPE)**: $\frac{1}{N}\sum \left|\frac{\hat{P}_i - P_i}{P_i}\right| \times 100\%$

#### Directional Edge (The True Trading Metric):
- **Directional Accuracy (Hit Rate %)**:
  $$\text{Hit Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\text{sign}(\hat{R}_{i}) == \text{sign}(R_{i})\right) \times 100\%$$
  A model with 55%+ directional hit rate on liquid equities has demonstrable statistical edge.

