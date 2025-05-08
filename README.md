# Martingale Property Testing in LLMs

This repository contains code to test the martingale property in language models. The experiments generate statistical distributions and evaluate how different LLMs maintain consistency in their probabilistic predictions.

## Project Overview

The code implements testing frameworks to evaluate if various LLMs (like GPT-4o-mini, Grok-beta, and DeepSeek-chat) satisfy the martingale property when generating samples from statistical distributions.

Two implementations are provided:
- `experiment.py`: Initial implementation using Together AI API (more complex and I couldn't make it work efficiently)
- `martingale_experiments.py`: Current implementation using OpenAI CLI for Grok, Deepseek, and 4o-mini

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv env
   source env/bin/activate 
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory with the following API keys:
```
TOGETHER_API_KEY=your_together_api_key  # For experiment.py
OPENAI_API_KEY=your_openai_api_key      
XAI_API_KEY=your_xai_api_key            
DEEPSEEK_API_KEY=your_deepseek_api_key   
```

## Running Experiments

To run the experiments:
```
python martingale_experiments.py
```

The script will:
1. Generate synthetic datasets (Bernoulli and Gaussian distributions)
2. Query different LLM models to generate sample paths
3. Calculate diagnostic statistics (T1 and T2)
4. Save results as JSON and generate plots

## Results

Results are saved in:
- `results/`: JSON files with full experiment data
- `plots/`: Visualization of T1 and T2 metrics across different parameters
