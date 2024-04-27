import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

from bert_train import BertForNextTokenPrediction, BertModel, preprocess_function, load_latest_checkpoint

def evaluate_model(model, data_loader, device):

    model.eval()
    total_loss = 0
    total_items = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids, labels = input_ids[:, :-1], input_ids[:, 1:]
            attention_mask = attention_mask[:, :-1]

            outputs = model(input_ids, attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            total_loss += loss.item() * input_ids.size(0)
            total_items += input_ids.size(0)

    avg_loss = total_loss / total_items
    print(f"Evaluation Ave Loss: {avg_loss:.2f}")

# params
vocab_size = 30522
hidden_size = 100
num_hidden_layers = 4
num_attention_heads = 4
intermediate_size = 200
max_position_embeddings = 128
max_length = 32
batch_size = 32
learning_rate = 5e-5
num_epochs = 100    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataloader = DataLoader(encoded_dataset, batch_size=32, shuffle=False)

    base_bert_model = BertModel(
        vocab_size, 
        hidden_size, 
        num_hidden_layers, 
        num_attention_heads, 
        intermediate_size, 
        max_position_embeddings).to(device)

    model = BertForNextTokenPrediction(base_bert_model, vocab_size).to(device)

    checkpoint = load_latest_checkpoint()  # Load the latest training checkpoint
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found, starting from scratch.")
        return

    evaluate_model(model, val_dataloader, device)

if __name__ == "__main__":
    main()

