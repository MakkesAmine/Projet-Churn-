name: MLOps Pipeline

on:
  push:
    branches:
      - main

jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

  run_pipeline:
    runs-on: ubuntu-latest
    needs: setup
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

      - name: Run ML pipeline
        run: |
          source venv/bin/activate
          python main.py --train_path data/churn-bigml-80.csv --test_path data/churn-bigml-20.csv --model_path models/churn_model.pkl

  clean:
    runs-on: ubuntu-latest
    needs: run_pipeline
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Clean temporary files
        run: |
          rm -rf data/processed/*
          rm -rf models/*

