# lora_finetune_jokes.py

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# -------------------------
# 1. Joke Dataset
# -------------------------
data = {
    "text": [
        "Why don’t scientists trust atoms? Because they make up everything!",
        "I told my computer I needed a break, and it froze.",
        "Why was the math book sad? Because it had too many problems.",
        "Parallel lines have so much in common. It’s a shame they’ll never meet.",
        "Why don’t programmers like nature? Too many bugs."
    ]
}
dataset = Dataset.from_dict(data)

# -------------------------
# 2. Load GPT-2
# -------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_name)

# -------------------------
# 3. LoRA Config
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# -------------------------
# 4. Preprocess Data
# -------------------------
def preprocess(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(preprocess)

# -------------------------
# 5. Training Setup
# -------------------------
training_args = TrainingArguments(
    output_dir="./lora-jokes",
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_dir="./logs",
    save_strategy="no",
    logging_steps=1,
    learning_rate=1e-4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# -------------------------
# 6. Train Model
# -------------------------
trainer.train()

# -------------------------
# 7. Save LoRA Adapter
# -------------------------
model.save_pretrained("./lora-jokes")

# -------------------------
# 8. Joke Generation
# -------------------------
def tell_joke(prompt="Why"):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    print("\nPrompt:", prompt)
    print("Joke:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# Try generating jokes
tell_joke("Why do programmers")
tell_joke("Why was the computer")
tell_joke("Knock knock")
