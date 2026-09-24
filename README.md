# PRISM GenAI Hackathon 3.0 — Agentic Code Intelligence

## Problem

Given a natural-language programming problem and a library of code snippets,
the system retrieves and ranks the most relevant snippets.

This project focuses on retrieval, not code generation.

## Our Approach

We use a Problem-Focused Multi-View Retrieval approach.

Each programming query is decomposed into three views:

- Core — main programming task
- Input — input requirements and constraints
- Output — expected result/output

Each view is encoded separately and combined using:

Core   = 0.50
Input  = 0.25
Output = 0.25

Final Query Vector =
0.50 × Core + 0.25 × Input + 0.25 × Output

The final vector is used to retrieve and rank code snippets.

## Model

`sentence-transformers/all-MiniLM-L6-v2`

The system is designed for CPU-oriented execution.

## Architecture

```text
Programming Query
       ↓
Query Preprocessing
       ↓
Core / Input / Output
       ↓
Three Embeddings
       ↓
Weighted Fusion
0.50 / 0.25 / 0.25
       ↓
Similarity Retrieval
       ↓
Ranked Code Snippets