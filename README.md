# \# PRISM GenAI Hackathon 3.0 — Agentic Code Intelligence

# 

# \## Problem

# 

# Given a natural-language programming problem and a library of code snippets, the system retrieves and ranks the most relevant snippets.

# 

# This project focuses on retrieval, not code generation.

# 

# \## Our Approach

# 

# We use a Problem-Focused Multi-View Retrieval approach.

# 

# Each programming query is decomposed into three views:

# 

# \- Core — main programming task

# \- Input — input requirements and constraints

# \- Output — expected result/output

# 

# Each view is encoded separately and combined using:

# 

# Core   = 0.50

# Input  = 0.25

# Output = 0.25

# 

# The final query vector is:

# 

# Final Query Vector =

# 0.50 × Core + 0.25 × Input + 0.25 × Output

# 

# The final vector is used to retrieve and rank code snippets.

# 

# \## Model

# 

# sentence-transformers/all-MiniLM-L6-v2

# 

# The system is designed for CPU-oriented execution.

# 

# \## Architecture

# 

# Programming Query

# &#x20;      ↓

# Query Preprocessing

# &#x20;      ↓

# Core / Input / Output

# &#x20;      ↓

# Three Embeddings

# &#x20;      ↓

# Weighted Fusion

# 0.50 / 0.25 / 0.25

# &#x20;      ↓

# Similarity Retrieval

# &#x20;      ↓

# Ranked Code Snippets

# 

# \## Version-Aware Retrieval

# 

# The system also supports version-aware indexing for evolving codebases.

# 

# Each code snippet is identified using its content hash. When a snippet remains unchanged between versions, its existing embedding is reused. When a snippet is added or modified, only the changed snippet is encoded again.

# 

# This allows incremental indexing while avoiding unnecessary recomputation.

# 

# \## Retrieval Examples

# 

# Example query:

# 

# How is the input validated?

# 

# The system retrieves the validate\_input function as a top-ranked result.

# 

# Another example query:

# 

# Which function converts spaces into underscores?

# 

# The system retrieves the normalize function as a top-ranked result.

# 

# \## Evaluation

# 

# The system was evaluated on the AppsRetrieval benchmark.

# 

# Our baseline run achieved an NDCG@10 of 0.0660.

# 

# The final multi-view retrieval method achieved an NDCG@10 of 0.07487.

# 

# The evaluation contains 3,765 test queries and 8,765 corpus snippets.

# 

# The version-aware indexing feature was evaluated separately using local versioned indexing tests.

# 

# \## Technology

# 

# \- Python

# \- Sentence Transformers

# \- ONNX Runtime

# \- Streamlit

# \- BM25 and dense retrieval

# \- CPU-oriented inference

# 

# \## Project Structure

# 

# project/

# │

# ├── app.py

# ├── retrieval\_worker.py

# ├── requirements.txt

# ├── Dockerfile

# │

# ├── src/

# │   ├── encoder.py

# │   ├── hybrid.py

# │   ├── preprocess.py

# │   ├── query\_views.py

# │   ├── versions.py

# │   └── versioned\_retrieval.py

# │

# ├── data/

# │   ├── sample\_snippets.jsonl

# │   └── versions/

# │       ├── v1.jsonl

# │       └── v2.jsonl

# │

# └── README.md

# 

# \## Running the Application

# 

# Activate the virtual environment:

# 

# .venv\\Scripts\\Activate.ps1

# 

# Start the Streamlit dashboard:

# 

# streamlit run app.py --server.fileWatcherType none

# 

# \## Version-Aware Indexing Demo

# 

# Build Version 1:

# 

# python src/versioned\_retrieval.py --version v1 --build

# 

# Rebuild Version 1:

# 

# python src/versioned\_retrieval.py --version v1 --build

# 

# Build Version 2:

# 

# python src/versioned\_retrieval.py --version v2 --build

# 

# Query Version 2:

# 

# python src/versioned\_retrieval.py --version v2 --query "Which function converts spaces into underscores?" --top-k 3

# 

# \## Demo Video

# 

# Watch the CodeLens-R Demo Video:

# 

# https://drive.google.com/file/d/1ADJ7gxi8Kh7vaYfwi-eDJA-nOZ8XqCWK/view?usp=sharing

# 

# \## Team

# 

# Team Rinita

# 

# Organization: VIT

# 

# Project: CodeLens-R — Version-Aware Multi-View Code Retrieval

