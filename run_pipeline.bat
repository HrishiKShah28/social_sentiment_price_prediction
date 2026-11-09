@echo off
echo ==============================================================================
echo Running HengSeng Sentiment Price Prediction Agent - End-to-End Pipeline
echo ==============================================================================
python -m src.pipeline.run --stage all
pause
