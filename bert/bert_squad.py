from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from tqdm import tqdm

# Load the SQuAD dataset
squad_dataset = load_dataset("squad")

# Prompt the user for the percentage of the dataset to use
train_percentage = 0.01
val_percentage = 0.01

# Calculate the number of examples based on the percentages
train_num_examples = int(len(squad_dataset["train"]) * train_percentage)
val_num_examples = int(len(squad_dataset["validation"]) * val_percentage)

# Select a subset of the dataset based on the percentages
train_dataset = squad_dataset["train"].select(range(train_num_examples))
val_dataset = squad_dataset["validation"].select(range(val_num_examples))

# Load the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Fine-tune the model with progress bar
print("Fine-tuning the model...")
train_results = trainer.train()
print("Fine-tuning completed.")

# Evaluate the model with progress bar
print("Evaluating the model...")
eval_results = trainer.evaluate()
print("Evaluation completed.")

# Print the training and evaluation results
print("Training results:", train_results)
print("Evaluation results:", eval_results)