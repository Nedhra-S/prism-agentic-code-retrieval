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

# \- \*\*Core\*\* — main programming task

# \- \*\*Input\*\* — input requirements and constraints

# \- \*\*Output\*\* — expected result/output

# 

# Each view is encoded separately and combined using:

# 

# ```text

# Core   = 0.50

# Input  = 0.25

# Output = 0.25

