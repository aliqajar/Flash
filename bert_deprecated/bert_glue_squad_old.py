import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BertForSequenceClassification
from datasets import load_dataset
from evaluate import load as load_metric
from torch.utils.data import DataLoader

def evaluate(model, dataset, device, task, tokenizer, batch_size=16):
    model.eval()
    model.to(device)
    metric = load_metric("glue", task) if task != "squad" else load_metric("squad")

    dataloader = DataLoader(dataset["validation"], batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            if task != "squad":
                inputs = tokenizer(batch["sentence1"], batch["sentence2"], return_tensors="pt", truncation=True, padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                references = batch["label"]
            else:
                inputs = tokenizer(batch["question"], batch["context"], padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                
                # Correcting data extraction from batch['answers']
                if 'answers' in batch and batch['answers']:
                    answer_starts = [ans['answer_start'][0] for ans in batch['answers']['answers'] if ans['answer_start']]
                    answer_texts = [ans['text'][0] for ans in batch['answers']['answers'] if ans['text']]
                    end_positions = [start + len(text) for start, text in zip(answer_starts, answer_texts)]
                else:
                    answer_starts = []
                    end_positions = []

                predictions = {'start_positions': outputs.start_logits, 'end_positions': outputs.end_logits}
                references = {'start_positions': torch.tensor(answer_starts, device=device), 'end_positions': torch.tensor(end_positions, device=device)}

            metric.add_batch(predictions=predictions, references=references)

    return metric.compute()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print("Device:", device)

    # GLUE tasks
    glue_datasets = load_dataset("glue", "mrpc")
    model = BertForSequenceClassification.from_pretrained(model_name)
    glue_results = evaluate(model, glue_datasets, device, "mrpc", tokenizer)
    print("GLUE Results:", glue_results)

    # SQuAD tasks
    squad_dataset = load_dataset("squad")
    model = BertForQuestionAnswering.from_pretrained(model_name)
    squad_results = evaluate(model, squad_dataset, device, "squad", tokenizer)
    print("SQuAD Results:", squad_results)
