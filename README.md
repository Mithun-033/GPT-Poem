# GPT-Poem

GPT-Poem is a decoder-only Transformer (GPT-style) trained to generate poetry from text prompts.
The project focuses on understanding, implementing, training, and running a language model
end-to-end using PyTorch.

---

## Overview

- Custom GPT-style architecture
- Trained on a poetry dataset
- Prompt-based text generation
- Local inference using PyTorch
- Educational and experimental focus

---

## Repository Structure

    GPT-Poem/
    ├── GPT_Inference.py
    ├── GPT_Complete.ipynb
    ├── Model_FineTuned.pt
    ├── Tokenizer (1).json
    ├── Datasets/
    ├── Models/
    └── README.md

---

## Setup

Clone the repository:

    git clone https://github.com/Mithun-033/GPT-Poem.git
    cd GPT-Poem

(Optional) Create a virtual environment:

    python -m venv venv
    source venv/bin/activate

Install required libraries (minimum):

    pip install torch transformers numpy

---

## Running Inference

Run the inference script:

    python GPT_Inference.py

Modify the prompt inside the script to control the generated poem.

---

## Notebook

The `GPT_Complete.ipynb` notebook contains:
- Model architecture
- Training logic
- Tokenization
- Sample generations

It is intended for experimentation and understanding the full pipeline.

---

## Notes

- This model is trained for creative generation, not factual accuracy
- Output quality depends heavily on prompt design
- Small model size limits coherence over long generations

---

## Future Work

- Better sampling strategies (temperature, top-k, top-p)
- Larger datasets
- Model scaling
- Cleaner CLI interface

---

## License

This project is provided for educational and research purposes.
