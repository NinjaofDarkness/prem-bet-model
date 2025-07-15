# ml-prem-cash-machine
Premier League 2025/26 table predictor

**Predicting the 2025/2026 Premier League table using a robust, monitorable, and scalable MLOps pipeline.**

## 🏆 Project Overview

This project demonstrates industry-grade MLOps principles for football analytics. It predicts the full Premier League table and is designed to be easy to scale, maintain, monitor, and extend—all with a focus on tools familiar to modern data teams.

## 🚀 Key Features

- End-to-end machine learning pipeline: data ingestion, feature engineering, training, evaluation, inference.
- All data stored in CSV/Parquet/DuckDB files (no SQL/Cloud required).
- Version-controlled, containerized, and fully automated (CI/CD-ready).
- Model and service monitoring with Prometheus & Grafana.
- API endpoints for predictions via FastAPI.
- Reproducible pipeline: plug-and-play with real or synthetic datasets.

## 💻 Tech Stack

- **Language:** Python (3.11+)
- **MLOps Tools:** Git, Docker, GitHub Actions
- **ML/DS:** pandas, scikit-learn, PyTorch (optional), DuckDB, Parquet
- **Pipeline Orchestration:** Airflow
- **Serving:** FastAPI
- **Monitoring:** Prometheus, Grafana
- **Testing:** pytest
