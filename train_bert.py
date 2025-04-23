from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch
import sys
from initialize import *
from common.logger import *

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    masked_positions = labels != -100
    correct = (predictions == labels) & masked_positions
    accuracy = correct.sum().item() / masked_positions.sum().item()
    return {"masked_lm_accuracy": accuracy}

def group_texts(examples):
    block_size = 128
    joined = {k: sum(examples[k], []) for k in examples}
    total_length = (len(joined["input_ids"]) // block_size) * block_size
    return {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in joined.items()
    }

def main():
    # for optimizer in [sgd, adamw_torch]:
    # for initialization_name in ["fernandez", "default"]:
    optimizer = "sgd"
    initialization_name = "fernandez"
    if initialization_name == "fernandez":
        initialization = fernandez_sinusoidal
    else:
        initialization = default_initialization  
    
        
    output_file = f"./results/outputwikitextbert-{initialization_name}-{optimizer}.log"
    sys.stdout = Logger(output_file)
    print(f"wikitext; bert; {initialization_name}; {optimizer}", flush=True)

    # Step 1: Load WikiText-2
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Step 2: Load pretrained tokenizer (from prajjwal1/bert-mini)
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-mini")

    # Step 3: Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # Step 4: Group into chunks
    lm_dataset = tokenized.map(group_texts, batched=True)

    # Step 5: Mask tokens dynamically
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Step 6: Create config from scratch (match prajjwal1/bert-mini)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id
    )

    # Step 7: Initialize model from config (no pretrained weights!)
    model = BertForMaskedLM(config)
    model.apply(initialization)  # Apply custom initialization

    # Step 8: Training arguments
    training_args = TrainingArguments(
        num_train_epochs=500,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="./results",
        logging_strategy="epoch",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        optim=optimizer,
        learning_rate=5e-5,
        weight_decay=0.001,
        lr_scheduler_type = "constant",
    )

    # Step 9: Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 10: Train from scratch!
    trainer.train()

if __name__ == "__main__":
    main()
