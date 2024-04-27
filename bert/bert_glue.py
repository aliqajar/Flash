import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load


def compute_metrics(eval_pred):
    metric_acc = load("accuracy")
    metric_f1 = load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="binary")["f1"]
    
    return {"accuracy": acc, "f1": f1}


def main(task="mrpc"):
    # Step 1: Setup — Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Determine the number of labels based on the task
    num_labels = 3 if task == "mnli" else 2
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Step 2: Load the specified GLUE dataset
    dataset = load_dataset("glue", task)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # Tokenization function
    def tokenize_function(examples):
        if task == "sst2":  # Single sentence tasks
            return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)
        else:  # Tasks with sentence pairs
            return tokenizer(examples['sentence1'], examples['sentence2'], padding="max_length", truncation=True, max_length=128)

    # Tokenize all data
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results/' + task,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs/' + task,
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
    print(f"Evaluation results for {task}:", results)

if __name__ == "__main__":
    # Example: run for MRPC
    main(task="mrpc")
    # main(task="mnli")
    # main(task="sst2")
    # main(task="qnli")
    # main(task="rte")
