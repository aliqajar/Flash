from my_bert_replica import BertTokenizer, BertModel, BertForNextTokenPrediction
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

# Load the dataset
dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")

# Parameters
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
max_length = 128
batch_size = 32

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_position_embeddings)
model = BertForNextTokenPrediction(model, vocab_size).to(device)

def preprocess_function(examples):
    texts = [text for text in examples["text"]]
    batch = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True)

# Example of encoding texts (no training)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids, attention_mask)
        # 'outputs' now contains the encoded representations of the input text

# Note: At this point, 'outputs' can be used for various downstream tasks or analyses.
