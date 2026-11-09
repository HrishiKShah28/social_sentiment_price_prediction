"""Interactive financial sentiment intelligence and prediction visual report."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import select, func
from src.config import settings
from src.storage.database import get_db_session, init_db
from src.storage.models import (
    Asset,
    Exchange,
    ModelFeature,
    ModelPrediction,
    NewsArticle,
    PriceBar,
    SocialPost,
    TechnicalLevel,
)
from src.storage.repository import DataRepository

st.set_page_config(
    page_title="HengSeng Sentiment Price Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_assets_summary(session, exchange_code: str = None):
    """Load latest ranking data for assets."""
    query = session.query(Asset).join(Asset.exchange)
    if exchange_code:
        query = query.filter(Exchange.code == exchange_code)
    assets = query.all()

    summary_rows = []
    for asset in assets:
        # Latest price
        latest_bar = (
            session.query(PriceBar)
            .filter_by(asset_id=asset.id)
            .order_by(PriceBar.timestamp_utc.desc())
            .first()
        )
        current_px = latest_bar.close if latest_bar else 0.0

        # Social mentions & sentiment
        social_count = session.query(func.count(SocialPost.id)).filter_by(asset_id=asset.id).scalar() or 0
        news_count = session.query(func.count(NewsArticle.id)).filter_by(asset_id=asset.id).scalar() or 0

        # Latest features / technical level
        latest_level = (
            session.query(TechnicalLevel)
            .filter_by(asset_id=asset.id)
            .order_by(TechnicalLevel.timestamp_utc.desc())
            .first()
        )
        latest_feat = (
            session.query(ModelFeature)
            .filter_by(asset_id=asset.id)
            .order_by(ModelFeature.timestamp_utc.desc())
            .first()
        )

        sentiment_score = latest_feat.composite_sentiment if latest_feat else 0.0
        direction = latest_level.trend_direction if latest_level else "Neutral"
        position = latest_level.suggested_position if latest_level else "Flat"
        support_1 = latest_level.support_1 if latest_level else 0.0
        resistance_1 = latest_level.resistance_1 if latest_level else 0.0

        # Latest prediction
        latest_pred = (
            session.query(ModelPrediction)
            .filter_by(asset_id=asset.id)
            .order_by(ModelPrediction.target_timestamp_utc.desc())
            .first()
        )
        pred_px = latest_pred.predicted_price if latest_pred else current_px

        summary_rows.append(
            {
                "Ticker": asset.ticker,
                "Company Name": asset.name,
                "Exchange": asset.exchange.code,
                "Current Price": current_px,
                "Predicted (n+1)": pred_px,
                "Social Mentions": social_count,
                "News Articles": news_count,
                "Sentiment Score": sentiment_score,
                "Direction": direction,
                "Suggested Position": position,
                "Support (S1)": support_1,
                "Resistance (R1)": resistance_1,
                "asset_id": asset.id,
            }
        )

    return pd.DataFrame(summary_rows)


def main():
    init_db()

    # Header
    st.title("📈 HengSeng Sentiment Price Prediction Agent")
    st.caption(
        "Institutional Alternative Data Sentiment Intelligence & Walk-Forward Price Forecaster | **CrowdWisdomTrading**"
    )

    with get_db_session() as session:
        repo = DataRepository(session)
        # Ensure default assets seeded
        repo.seed_exchanges_and_assets()

        # Sidebar navigation & filters
        st.sidebar.header("🎯 Navigation & Controls")
        exchange_filter = st.sidebar.selectbox(
            "Select Exchange Hub",
            options=["All Exchanges", "Hong Kong (HKEX / Hang Seng)", "South Korea (KRX / KOSPI)"],
            index=0,
        )

        code_map = {
            "Hong Kong (HKEX / Hang Seng)": "HKEX",
            "South Korea (KRX / KOSPI)": "KRX",
            "All Exchanges": None,
        }
        selected_code = code_map[exchange_filter]

        st.sidebar.markdown("---")
        st.sidebar.subheader("API Status & Provenance")
        apify_status = "🟢 Configured" if settings.APIFY_API_TOKEN else "🟡 Using Cached / Seed"
        exa_status = "🟢 Configured" if settings.EXA_API_KEY else "🟡 Using Cached / Seed"
        st.sidebar.write(f"**Apify Actors**: {apify_status}")
        st.sidebar.write(f"**Exa News**: {exa_status}")
        st.sidebar.write(f"**Price Data**: 🟢 yfinance Live")

        # Load ranking data
        df_summary = load_assets_summary(session, exchange_code=selected_code)

        if df_summary.empty:
            st.warning("No asset data found in database. Please run the ingestion pipeline.")
            return

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets Tracked", len(df_summary))
        total_posts = int(df_summary["Social Mentions"].sum())
        col2.metric("Total Social Posts", total_posts)
        total_news = int(df_summary["News Articles"].sum())
        col3.metric("Total Financial News", total_news)
        avg_sentiment = float(df_summary["Sentiment Score"].mean())
        col4.metric("Avg Sentiment Score", f"{avg_sentiment:+.3f}")

        # Tabbed Layout
        tab_rankings, tab_asset_deepdive, tab_validation, tab_provenance = st.tabs(
            [
                "🏆 Asset Rankings & Signals",
                "📊 Candlestick & Technical Levels",
                "🧠 ML Out-of-Sample Predictions",
                "📜 Raw Provenance Feed",
            ]
        )

        # TAB 1: RANKING REPORT
        with tab_rankings:
            st.subheader("Asset Rankings by Social Buzz, Sentiment & Direction")
            st.markdown(
                "Rank and evaluate assets dynamically across Hong Kong and South Korean exchanges based on retail buzz volume, multi-source sentiment scores, and technical breakout posture."
            )

            sort_by = st.selectbox(
                "Sort Table By:",
                options=["Social Mentions", "Sentiment Score", "Current Price", "Company Name"],
                index=0,
            )
            df_sorted = df_summary.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

            # Styling helpers
            def color_direction(val):
                if val == "Bullish":
                    return "color: #10B981; font-weight: bold;"
                elif val == "Bearish":
                    return "color: #EF4444; font-weight: bold;"
                return "color: #9CA3AF;"

            display_cols = [
                "Ticker",
                "Company Name",
                "Exchange",
                "Current Price",
                "Predicted (n+1)",
                "Social Mentions",
                "News Articles",
                "Sentiment Score",
                "Direction",
                "Suggested Position",
                "Support (S1)",
                "Resistance (R1)",
            ]
            st.dataframe(
                df_sorted[display_cols].style.applymap(color_direction, subset=["Direction", "Suggested Position"]),
                use_container_width=True,
                height=380,
            )

        # TAB 2: CANDLESTICK & TECHNICAL LEVELS
        with tab_asset_deepdive:
            st.subheader("Interactive Price Candlesticks with Support / Resistance Overlays")
            selected_ticker = st.selectbox("Select Asset to Inspect:", options=df_summary["Ticker"].tolist())
            asset_rec = repo.get_asset_by_ticker(selected_ticker)

            if asset_rec:
                bars = repo.get_price_bars(asset_rec.id)
                if bars:
                    df_bars = pd.DataFrame(
                        [
                            {
                                "date": b.timestamp_utc,
                                "open": b.open,
                                "high": b.high,
                                "low": b.low,
                                "close": b.close,
                                "volume": b.volume,
                            }
                            for b in bars
                        ]
                    )

                    latest_lvl = (
                        session.query(TechnicalLevel)
                        .filter_by(asset_id=asset_rec.id)
                        .order_by(TechnicalLevel.timestamp_utc.desc())
                        .first()
                    )

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{asset_rec.ticker} — Price Bars & Support/Resistance Levels", "Trading Volume"),
                    )

                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=df_bars["date"],
                            open=df_bars["open"],
                            high=df_bars["high"],
                            low=df_bars["low"],
                            close=df_bars["close"],
                            name="OHLC",
                        ),
                        row=1,
                        col=1,
                    )

                    # Support and Resistance horizontal markers
                    if latest_lvl:
                        fig.add_hline(
                            y=latest_lvl.resistance_1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"R1: {latest_lvl.resistance_1}",
                            row=1,
                            col=1,
                        )
                        fig.add_hline(
                            y=latest_lvl.support_1,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"S1: {latest_lvl.support_1}",
                            row=1,
                            col=1,
                        )

                    # Volume bar
                    fig.add_trace(
                        go.Bar(x=df_bars["date"], y=df_bars["volume"], name="Volume", marker_color="rgba(100, 150, 250, 0.6)"),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No price bars ingested yet for {selected_ticker}.")

        # TAB 3: MODEL PREDICTIONS & OUT-OF-SAMPLE VALIDATION
        with tab_validation:
            st.subheader("Machine Learning Out-of-Sample Walk-Forward Validation")
            st.markdown(
                "Predictions evaluated chronologically with expanding windows and embargo to guarantee zero lookahead leakage."
            )

            preds_query = (
                session.query(ModelPrediction)
                .join(ModelPrediction.asset)
                .order_by(ModelPrediction.target_timestamp_utc.desc())
                .all()
            )

            if preds_query:
                df_preds = pd.DataFrame(
                    [
                        {
                            "Ticker": p.asset.ticker,
                            "Model": p.model_name,
                            "Cutoff Time": p.feature_cutoff_utc,
                            "Base Price": p.base_price,
                            "Predicted Return": f"{p.predicted_return * 100:+.2f}%",
                            "Predicted Price": p.predicted_price,
                            "Actual Price": p.actual_price,
                            "Abs Error": p.error_abs,
                            "Hit Rate Status": "✅ Correct" if p.directional_correct else "❌ Incorrect",
                        }
                        for p in preds_query
                    ]
                )
                st.dataframe(df_preds, use_container_width=True, height=400)
            else:
                st.info("Run `python -m src.pipeline.run --stage train` to generate walk-forward validation predictions.")

        # TAB 4: RAW PROVENANCE FEED
        with tab_provenance:
            st.subheader("Verified Data Provenance: Scraped Social Posts & Exa News Articles")
            provenance_asset = st.selectbox("Inspect Provenance for Asset:", options=df_summary["Ticker"].tolist(), key="prov_ticker")
            target_asset = repo.get_asset_by_ticker(provenance_asset)

            if target_asset:
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.markdown("#### Real Scraped Social Posts (Apify)")
                    posts = (
                        session.query(SocialPost)
                        .filter_by(asset_id=target_asset.id)
                        .order_by(SocialPost.timestamp_utc.desc())
                        .limit(20)
                        .all()
                    )
                    if posts:
                        for post in posts:
                            with st.expander(f"[{post.platform}] {post.timestamp_utc.strftime('%Y-%m-%d %H:%M')} UTC | 👍 {post.likes_count} | 👁️ {post.views_count}"):
                                st.write(post.post_text)
                                st.caption(f"Author: {post.author_id or 'Anonymous'} (Followers: {post.author_followers}) | URL: {post.post_url or 'N/A'}")
                    else:
                        st.caption("No social posts ingested yet.")

                with p_col2:
                    st.markdown("#### Real Financial News (Exa.ai)")
                    articles = (
                        session.query(NewsArticle)
                        .filter_by(asset_id=target_asset.id)
                        .order_by(NewsArticle.published_date_utc.desc())
                        .limit(20)
                        .all()
                    )
                    if articles:
                        for art in articles:
                            with st.expander(f"[{art.domain or 'News'}] {art.title[:65]}..."):
                                st.write(art.snippet or art.title)
                                st.markdown(f"[Read full article]({art.url})")
                                st.caption(f"Published: {art.published_date_utc.strftime('%Y-%m-%d %H:%M')} UTC | Relevance Score: {art.exa_score}")
                    else:
                        st.caption("No financial news articles ingested yet.")


if __name__ == "__main__":
    main()
