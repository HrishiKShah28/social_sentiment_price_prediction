"""End-to-end pipeline orchestrator executing ingestion, intelligence, and modeling."""

from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy import select
from src.config import settings
from src.features.builder import FeatureBuilder
from src.ingestion.apify_client import ApifySocialService
from src.ingestion.exa_client import ExaNewsService
from src.ingestion.price_client import PriceDataService
from src.models.validation import WalkForwardValidator
from src.nlp.levels import TechnicalLevelExtractor
from src.nlp.sentiment import SentimentScorer
from src.storage.database import get_db_session, init_db
from src.storage.models import (
    Asset,
    ModelFeature,
    ModelPrediction,
    NewsArticle,
    PriceBar,
    SentimentScore,
    SocialPost,
    TechnicalLevel,
)
from src.storage.repository import DataRepository


class PipelineOrchestrator:
    """Master controller managing data ingestion, feature generation, and ML training."""

    def __init__(self):
        init_db()
        self.scorer = SentimentScorer()
        self.feature_builder = FeatureBuilder(self.scorer)

    def run_ingestion_stage(self) -> Dict[str, int]:
        """Execute market price, news, and social data ingestion for all assets."""
        logger.info("=== Starting Ingestion Stage ===")
        metrics = {"price_bars": 0, "news_articles": 0, "social_posts": 0}

        with get_db_session() as session:
            repo = DataRepository(session)
            repo.seed_exchanges_and_assets()
            assets = repo.get_active_assets()

            price_service = PriceDataService(repo)
            exa_service = ExaNewsService(repo)
            apify_service = ApifySocialService(repo)

            for asset in assets:
                # 1. Market Price Bars
                try:
                    bars_count = price_service.fetch_and_store_bars(asset)
                    metrics["price_bars"] += bars_count
                except Exception as e:
                    logger.error(f"Price ingestion failed for {asset.ticker}: {e}")

                # 2. Exa Financial News
                try:
                    news_count = exa_service.fetch_and_store_news(asset)
                    metrics["news_articles"] += news_count
                except Exception as e:
                    logger.error(f"Exa news ingestion failed for {asset.ticker}: {e}")

                # 3. Apify Social Posts
                # If APIFY_API_TOKEN is available, run live actor calls
                # Otherwise, graceful fallback is logged by ApifySocialService
                try:
                    if "HK" in asset.ticker:
                        apify_service.run_actor_for_asset(
                            asset=asset,
                            platform="LIHKG",
                            actor_id=settings.APIFY_LIHKG_ACTOR_ID,
                            run_input={"search": asset.ticker.split(".")[0], "limit": 20},
                        )
                    else:
                        apify_service.run_actor_for_asset(
                            asset=asset,
                            platform="Naver",
                            actor_id=settings.APIFY_NAVER_ACTOR_ID,
                            run_input={"code": asset.ticker.split(".")[0], "limit": 20},
                        )
                except Exception as e:
                    logger.error(f"Apify social ingestion failed for {asset.ticker}: {e}")

        logger.info(f"Ingestion completed: {metrics}")
        return metrics

    def run_processing_stage(self) -> Dict[str, int]:
        """Compute sentiment scores, technical levels, and point-in-time features."""
        logger.info("=== Starting Processing & Intelligence Stage ===")
        metrics = {"scored_items": 0, "technical_levels": 0, "features_generated": 0}

        with get_db_session() as session:
            repo = DataRepository(session)
            assets = repo.get_active_assets()

            for asset in assets:
                # 1. Fetch raw items
                bars = repo.get_price_bars(asset.id)
                posts = session.query(SocialPost).filter_by(asset_id=asset.id).all()
                news = session.query(NewsArticle).filter_by(asset_id=asset.id).all()

                # Score unscored social posts
                for p in posts:
                    res = self.scorer.score_text(p.post_text)
                    w = self.scorer.compute_engagement_weight(p.likes_count, p.reposts_count, p.views_count, p.author_followers)
                    score_rec = SentimentScore(
                        source_type="social_post",
                        source_id=p.id,
                        model_name=self.scorer.model_name,
                        raw_score=res.raw_score,
                        confidence=res.confidence,
                        label=res.label,
                        engagement_weight=w,
                    )
                    session.add(score_rec)
                    metrics["scored_items"] += 1

                # 2. Derive technical levels & position
                if bars and len(bars) >= 3:
                    # Compute composite sentiment for latest window
                    recent_posts = [p for p in posts if (datetime.utcnow() - p.timestamp_utc).days <= 3]
                    social_score = (
                        sum(self.scorer.score_text(p.post_text).raw_score for p in recent_posts) / len(recent_posts)
                        if recent_posts else 0.0
                    )
                    recent_news = [n for n in news if (datetime.utcnow() - n.published_date_utc).days <= 3]
                    news_score = (
                        sum(self.scorer.score_text(n.title).raw_score for n in recent_news) / len(recent_news)
                        if recent_news else 0.0
                    )
                    composite = self.scorer.compute_composite_sentiment({"recent_social": social_score}, news_score)

                    tech_signal = TechnicalLevelExtractor.extract_levels(bars, composite_sentiment=composite)
                    if tech_signal:
                        level_rec = TechnicalLevel(
                            asset_id=asset.id,
                            timestamp_utc=bars[-1].timestamp_utc,
                            current_price=tech_signal.current_price,
                            support_1=tech_signal.support_1,
                            support_2=tech_signal.support_2,
                            resistance_1=tech_signal.resistance_1,
                            resistance_2=tech_signal.resistance_2,
                            trend_direction=tech_signal.trend_direction,
                            suggested_position=tech_signal.suggested_position,
                            calculation_method=tech_signal.calculation_method,
                        )
                        session.add(level_rec)
                        metrics["technical_levels"] += 1

                # 3. Construct point-in-time features
                if bars:
                    feat_rows = self.feature_builder.build_features(bars, posts, news)
                    for f in feat_rows:
                        # Check if feature already saved for this asset and timestamp
                        existing = session.query(ModelFeature).filter_by(
                            asset_id=asset.id, timestamp_utc=f["timestamp_utc"]
                        ).first()
                        if not existing:
                            feat_rec = ModelFeature(
                                asset_id=asset.id,
                                timestamp_utc=f["timestamp_utc"],
                                timeframe=f["timeframe"],
                                social_mentions_count=f["social_mentions_count"],
                                social_sentiment_mean=f["social_sentiment_mean"],
                                social_sentiment_weighted=f["social_sentiment_weighted"],
                                social_sentiment_momentum=f["social_sentiment_momentum"],
                                news_articles_count=f["news_articles_count"],
                                news_sentiment_mean=f["news_sentiment_mean"],
                                composite_sentiment=f["composite_sentiment"],
                                return_lag1=f["return_lag1"],
                                return_lag5=f["return_lag5"],
                                rsi_14=f["rsi_14"],
                                macd_hist=f["macd_hist"],
                                atr_14=f["atr_14"],
                                hist_vol_10=f["hist_vol_10"],
                                volume_ratio=f["volume_ratio"],
                                sentiment_price_divergence=f["sentiment_price_divergence"],
                                target_next_return=f["target_next_return"],
                                target_next_price=f["target_next_price"],
                            )
                            session.add(feat_rec)
                            metrics["features_generated"] += 1

            session.commit()

        logger.info(f"Processing completed: {metrics}")
        return metrics

    def run_training_stage(self) -> Dict[str, Dict]:
        """Execute walk-forward cross-validation and save model forecasts."""
        logger.info("=== Starting Model Training & Walk-Forward Validation Stage ===")
        all_metrics: Dict[str, Dict] = {}

        with get_db_session() as session:
            repo = DataRepository(session)
            run = repo.start_pipeline_run(stage="train")
            assets = repo.get_active_assets()

            for asset in assets:
                feature_recs = (
                    session.query(ModelFeature)
                    .filter_by(asset_id=asset.id)
                    .order_by(ModelFeature.timestamp_utc.asc())
                    .all()
                )

                if len(feature_recs) < 6:
                    logger.warning(f"Not enough features to validate {asset.ticker} ({len(feature_recs)} rows)")
                    continue

                # Convert ORM features to dicts
                feature_rows = []
                for f in feature_recs:
                    feature_rows.append(
                        {
                            "timestamp_utc": f.timestamp_utc,
                            "current_price": float(
                                session.query(PriceBar)
                                .filter_by(asset_id=asset.id, timestamp_utc=f.timestamp_utc)
                                .first()
                                .close
                            )
                            if session.query(PriceBar).filter_by(asset_id=asset.id, timestamp_utc=f.timestamp_utc).first()
                            else 100.0,
                            "social_mentions_count": f.social_mentions_count,
                            "social_sentiment_mean": f.social_sentiment_mean,
                            "social_sentiment_weighted": f.social_sentiment_weighted,
                            "social_sentiment_momentum": f.social_sentiment_momentum,
                            "news_articles_count": f.news_articles_count,
                            "news_sentiment_mean": f.news_sentiment_mean,
                            "composite_sentiment": f.composite_sentiment,
                            "return_lag1": f.return_lag1,
                            "return_lag5": f.return_lag5,
                            "rsi_14": f.rsi_14,
                            "macd_hist": f.macd_hist,
                            "atr_14": f.atr_14,
                            "hist_vol_10": f.hist_vol_10,
                            "volume_ratio": f.volume_ratio,
                            "sentiment_price_divergence": f.sentiment_price_divergence,
                            "target_next_return": f.target_next_return,
                            "target_next_price": f.target_next_price,
                        }
                    )

                results = WalkForwardValidator.evaluate(feature_rows, min_train_size=min(5, len(feature_rows) - 2))

                for model_name, (metrics_obj, pred_records) in results.items():
                    for pred in pred_records:
                        pred_rec = ModelPrediction(
                            pipeline_run_id=run.id,
                            asset_id=asset.id,
                            model_name=model_name,
                            model_version="1.0.0",
                            feature_cutoff_utc=pred["timestamp_utc"],
                            target_timestamp_utc=pred["timestamp_utc"],
                            base_price=pred["base_price"],
                            predicted_return=pred["predicted_return"],
                            predicted_price=pred["predicted_price"],
                            predicted_direction="Bullish" if pred["predicted_return"] > 0 else "Bearish",
                            actual_price=pred["actual_price"],
                            actual_direction="Bullish" if pred["actual_return"] > 0 else "Bearish",
                            error_abs=pred["error_abs"],
                            error_pct=round(abs(pred["predicted_price"] - pred["actual_price"]) / pred["actual_price"] * 100, 2)
                            if pred["actual_price"] > 0 else 0.0,
                            directional_correct=pred["directional_correct"],
                        )
                        session.add(pred_rec)

                    all_metrics[f"{asset.ticker}_{model_name}"] = {
                        "rmse": metrics_obj.rmse,
                        "mae": metrics_obj.mae,
                        "mape": metrics_obj.mape,
                        "hit_rate": metrics_obj.directional_accuracy,
                    }

            repo.finish_pipeline_run(run.run_id, status="completed", metrics=all_metrics)

        logger.info(f"Training completed across {len(all_metrics)} model evaluations")
        return all_metrics

    def run_all(self) -> Dict[str, any]:
        """Run the full end-to-end pipeline: ingest -> process -> train."""
        ingest_res = self.run_ingestion_stage()
        process_res = self.run_processing_stage()
        train_res = self.run_training_stage()
        return {
            "ingestion": ingest_res,
            "processing": process_res,
            "training": train_res,
        }
