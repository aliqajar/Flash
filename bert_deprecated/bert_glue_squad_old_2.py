import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import pipeline
from tqdm import tqdm



def evaluate_glue(model, tokenizer, device, dataset):
    """Function to evaluate the model on GLUE dataset."""
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    model.to(device)

    total, correct = 0, 0
    for batch in dataloader:
        inputs = tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch['label'].to(device)).sum().item()
        total += len(batch['label'])

    return correct / total



def evaluate_squad(model, tokenizer, device, dataset):
    """Function to evaluate the model on SQuAD dataset."""
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device.index)
    total, correct = 0, 0

    for example in tqdm(dataset, desc="Evaluating SQuAD", unit=" example"):
        question, context = example['question'], example['context']
        correct_answer = example['answers']['text'][0]
        result = qa_pipeline({'question': question, 'context': context})
        predicted_answer = result['answer']
        correct += (predicted_answer.lower().strip() == correct_answer.lower().strip())
        total += 1

    return correct / total



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_glue = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model_squad = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    # Load the datasets
    glue_dataset = load_dataset("glue", "mrpc", split='validation')
    squad_dataset = load_dataset("squad", split='validation').select(range(100))  # Evaluate on a subset for quick testing

    # Evaluate on GLUE
    glue_accuracy = evaluate_glue(model_glue, tokenizer, device, glue_dataset)
    print(f"GLUE MRPC Accuracy: {glue_accuracy:.2f}")

    # Evaluate on SQuAD
    squad_accuracy = evaluate_squad(model_squad, tokenizer, device, squad_dataset)
    print(f"SQuAD Accuracy: {squad_accuracy:.2f}")



if __name__ == "__main__":
    main()
