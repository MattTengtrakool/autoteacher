import os
import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load the prompts and responses from the text file
input_file = "prompts_and_responsesnew.txt"

with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

# Create a list of examples
examples = content.split("\n\n")
examples = [example.replace("Prompt: ", "").replace("Response: ", "") for example in examples]

# Create a Dataset object from the list of examples
dataset = Dataset.from_dict({"text": examples})
print("Dataset created.")

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("GPT-2 tokenizer and model loaded.")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
print("Dataset tokenized.")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="output_dir",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=500,
    logging_dir="logging_dir",
)

# Set up a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()
print("Finished fine-tuning.")

# Save the fine-tuned model
trainer.save_model("fine_tuned_gpt2")
print("Model saved.")
