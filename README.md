# WeightLens

**Automated Interpretability for Large Language Models Using Weight-Based Feature Analysis**

WeightLens provides a low-cost framework for analyzing the meaning of LLM features using **input-invariant components** (weights) instead of repeated activation probing. By leveraging **transcoders**, it reduces reliance on external LLM explainers while producing meaningful interpretations for token-level features.

---

## Features

- **Feature-to-feature contribution analysis** across model layers
- **Weight-based interpretability** using transcoders (input-invariant)
- **Caching and JSON-based results storage** to avoid redundant computation
- **Human-readable feature descriptions** printed alongside contribution scores

---

## Installation

```bash
# Clone the repository
git clone https://github.com/egolimblevskaia/WeightLens.git
cd WeightLens

# Install dependencies
pip install -r requirements.txt
```

## Usage

Please refer to the [`example.ipynb`](example.ipynb) notebook for a complete walkthrough of:

- Loading models with transcoders
- Loading and saving precomputed analysis from huggingface
- Analyzing feature-to-feature contributions
- Visualizing and interpreting results

All examples and instructions are provided there.

---

## Precomputed Data

To facilitate quick experimentation, precomputed feature analyses are available for the following models:

- **Gemma**: [google/gemma-2-2b](https://huggingface.co/datasets/egolimblevskaia/weightlens-gemma-2-2b-transcoder-descriptions)
- **Llama**: [meta-llama/Llama-3.2-1B](https://huggingface.co/datasets/egolimblevskaia/weightlens-Llama-3.2-1B-transcoder-descriptions)
- **GPT-2**: [gpt2](https://huggingface.co/datasets/egolimblevskaia/weightlens-gpt2-transcoder-descriptions)

These datasets contain feature analyses across multiple layers and can be used directly without recomputation.

## Citation

```
@misc{weightlens,
  author = {Golimblevskaia, Elena and Puri, Bruno and Jain, Aakriti and Samek, Wojciech and Lapuschkin, Sebastian},
  title  = {WeightLens: Input-Independent Interpretability for LLM Transcoders},
  year   = {2025},
  url    = {https://github.com/egolimblevskaia/WeightLens}
}
```

