import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')  # binary for MRPC
    return {
        'accuracy': acc,
        'f1': f1
    }

def main():
    # Step 1: Setup — Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Step 2: Load the MRPC dataset from GLUE
    dataset = load_dataset("glue", "mrpc")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True, max_length=128)

    # Tokenize all data
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Step 4: Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Add compute_metrics to Trainer
    )

    # Step 5: Train the model
    trainer.train()

    # Step 6: Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
