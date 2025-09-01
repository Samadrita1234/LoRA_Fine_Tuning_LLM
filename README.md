# LoRA_Fine_Tuning_LLM

LoRA Fine-Tuning with GPT-2 (Joke Generator)

This project demonstrates Low-Rank Adaptation (LoRA) fine-tuning using Hugging Face’s Transformers and PEFT library.
It fine-tunes a GPT-2 model on a small joke dataset to generate funny one-liners.

🚀 Features

Custom toy dataset of jokes.

LoRA parameter-efficient fine-tuning (r=8, lora_alpha=16).

Uses Hugging Face Trainer for training.

Saves LoRA adapter for reuse.

Joke generation with temperature + top-p sampling.

Install dependencies
pip install torch transformers datasets peft

Training Details

Base model: gpt2

Epochs: 10

Batch size: 2

Learning rate: 1e-4

LoRA Config:

Rank (r): 8

Alpha: 16

Target modules: c_attn (GPT-2 attention)

Prompt: Why do programmers
Joke: Why do programmers hate nature? Because it has too many bugs!

Prompt: Knock knock
Joke: Knock knock. Who’s there? A byte. A byte who? A byte you can’t handle!


Dropout: 0.05

🤡 Example Outputs
