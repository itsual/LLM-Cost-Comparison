# LLM Cost Comparison Simulator

This repository contains a **Streamlit** application for comparing the 5-year total costs of:

- **Gemini 2.0 Flash (API-based)**  
- **Self-Hosted LLaMA on Google Cloud Platform (GCP)**

You can **customize** the parameters—like monthly tokens, usage growth, GPU costs, and labor—to see how each setup’s costs evolve over time. The app provides a year-by-year breakdown, calculates overall totals, and visualizes cumulative costs with a line chart.

APp Link --> [LLM Cost Simulation Tool](https://llm-cost-comparison-by-al.streamlit.app/)
---

## Features

1. **Gemini 2.0 Flash Pricing**  
   - Separate **input** and **output** token costs (default values from [the Gemini docs](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.0-flash)).
   - Control up to **5 decimal places** for precision.
   - Annual growth rate for monthly tokens.

2. **Self-Hosted LLaMA (GCP)**  
   - Monthly GPU costs (on-demand; e.g., A100/T4).
   - Optional extra GPU usage in the first year for fine-tuning.
   - Storage, networking, labor costs.
   - Incremental overhead each year.

3. **Interactive Sliders & Inputs**  
   - Adjust usage, costs, labor in the **sidebar**.
   - Real-time updates to the cost tables and chart.

4. **Comparison & Visualization**  
   - Two cost tables: **Gemini** vs. **LLaMA**.
   - Yearly and cumulative summaries.
   - Line chart for cumulative 5-year costs.

5. **“Show Assumptions” Button**  
   - Click to view the cost-model logic, references, and disclaimers.

---

## Table of Contents

- [Installation](#installation)
- [Disclaimer](#disclaimer)

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/llm-cost-comparison.git
   cd llm-cost-comparison
---
## Disclaimer
- This application is for educational and demonstration purposes only.
- Real-world pricing may differ based on region, usage tiers, or negotiated rates.
- Always verify costs with your cloud provider and official documentation.
- You can adjust these token costs for any proprietary model (Gemini, LLaMA, or others) as needed.
