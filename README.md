# LLM Simple

A small, educational, character-level language model built from scratch with PyTorch.

The project implements a decoder-style Transformer that learns to predict the next character in a sequence. It includes scripts for preparing a Portuguese Wikipedia dataset, training and validating the model, saving checkpoints, loading trained weights, and generating text.

## Features

- Character-level tokenizer
- Causal multi-head self-attention
- Token and positional embeddings
- Stacked Transformer blocks
- Next-character prediction with cross-entropy loss
- Training and validation metrics
- Automatic CPU or CUDA device selection
- Model checkpoint saving and loading
- Temperature-controlled text generation
- Streaming dataset preparation with a deterministic train/validation split

## Project Structure

```text
LLM-simple/
├── Datasets/
│   └── cria_txt-treino-teste.py   # Downloads and splits the dataset
├── modelo/
│   ├── gera.py                     # Loads a checkpoint and generates text
│   ├── modelo.py                   # Transformer architecture and training menu
│   ├── train.py                    # Training, validation, save, and load functions
│   ├── tratamento.py               # Tokenizer and batch creation
│   └── pesos/
│       └── modelo.pt               # Created after training
└── README.md
```

## Requirements

- Python 3.10 or newer
- PyTorch
- tqdm
- Hugging Face `datasets`

A CUDA-compatible GPU is optional, but strongly recommended for larger model configurations.

## Installation

Clone the repository:

```bash
git clone https://github.com/Rafaelgm04/LLM-simple.git
cd LLM-simple
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install torch tqdm datasets
```

## Preparing the Dataset

The dataset script streams Portuguese Wikipedia from `TucanoBR/wikipedia-PT`, collects approximately 20 million characters, and creates a deterministic split of roughly 90% training data and 10% validation data.

Run:

```bash
cd Datasets
python cria_txt-treino-teste.py
```

This creates:

```text
Datasets/treino.txt
Datasets/validacao.txt
```

The text files must be longer than the block size used during training and validation.

## Training the Model

Run the training interface from the `modelo` directory so the relative dataset paths resolve correctly:

```bash
cd ../modelo
python modelo.py
```

The menu provides the following options:

```text
1 - Train the model
2 - View the latest loss and accuracy values
3 - Validate the model
4 - Load a saved model
0 - Exit
```

When training is selected, enter a positive number of epochs. The model is saved automatically:

- Every 1,000 epochs
- At the end of the training run

The checkpoint is written to:

```text
modelo/pesos/modelo.pt
```

The checkpoint contains:

- Model weights
- Tokenizer vocabulary
- Model configuration

## Generating Text

After training a model, run:

```bash
python gera.py
```

Enter `1`, then provide a prompt. The script loads `pesos/modelo.pt` and generates 300 additional tokens using a default temperature of `0.8`.

The prompt may only contain characters that exist in the training and validation vocabulary. An unknown character will raise a tokenizer error.

## Temperature

Temperature controls the randomness of generated text:

- Lower values, such as `0.3`, make the output more predictable and repetitive.
- Values near `1.0` provide a balance between consistency and variety.
- Higher values, such as `1.5`, make the output more random and less coherent.

Temperature must be greater than zero.

## Default Model Configuration

The current configuration in `modelo/modelo.py` is:

| Parameter | Value | Description |
|---|---:|---|
| Block size | 128 | Maximum context length accepted by the model |
| Embedding size | 256 | Size of token and position vectors |
| Attention heads | 16 | Number of parallel attention heads |
| Transformer layers | 12 | Number of stacked Transformer blocks |
| Dropout | 0.3 | Regularization applied during training |
| Learning rate | 0.0003 | AdamW optimizer learning rate |
| Batch size | 16 | Number of sequences sampled per training step |

The embedding size must be divisible by the number of attention heads.

## Architecture

The model follows a decoder-only Transformer design:

1. Characters are converted into integer token IDs.
2. Token embeddings are combined with learned positional embeddings.
3. The sequence passes through multiple Transformer blocks.
4. Each block uses pre-layer normalization, causal multi-head attention, residual connections, and a GELU feed-forward network.
5. A final linear layer produces logits for every character in the vocabulary.
6. Cross-entropy loss trains the model to predict the next character.

The causal attention mask prevents each position from seeing future characters.

## Training and Validation Metrics

The project reports:

- **Loss:** cross-entropy error for next-character prediction. Lower is generally better.
- **Accuracy:** percentage of positions where the most likely predicted character matches the target character. Higher is generally better.

Accuracy should be interpreted together with loss and generated samples. A character-level model can obtain reasonable accuracy while still producing repetitive or incoherent text.

## Customization

The main architecture settings can be changed in `modelo/modelo.py`:

```python
tamanho_bloco = 128
tamanho_embedding = 256
numero_cabecas = 16
numero_camadas = 12
dropout = 0.3
```

Training defaults can be changed in `modelo/train.py`:

```python
tamanho_bloco = 64
tamanho_lote = 16
taxa_aprendizado = 3e-4
```

For consistent experiments, keep the model block size and the training/validation block size aligned.

## Current Limitations

- The tokenizer operates on individual characters rather than subword tokens.
- Unknown characters in prompts are not supported.
- The project does not currently provide a command-line argument interface.
- Training state for the optimizer is not stored in the checkpoint.
- Text quality depends heavily on dataset size, model size, training duration, and hardware.
- The default 12-layer configuration may be slow on a CPU.

## Educational Purpose

This repository is intended as a learning project for understanding tokenization, causal attention, Transformer blocks, language-model training, validation, checkpointing, and autoregressive text generation with PyTorch.
