#Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import time
import psutil
import os

#Device Configuration  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Shared Parameters Across Models 
vocab_size = 10000
max_length = 100
batch_size = 32
num_epochs = 10
learning_rate = 0.001

#Bert Model 
bert_model = 'prajjwal1/bert-tiny'

#Load and Train-Test Split Data: When testing out the 3 datasets, merely edit which CVS is being read 
df = pd.read_csv('WELFake_sub_dataset_3.csv')
full_article = df["title"] + " " + df["text"]
labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(full_article, labels, test_size=0.2, random_state=42, stratify=labels)

#Tokenizer from Bert Model 
tokenizer = AutoTokenizer.from_pretrained(bert_model)

#Prepare for Dataloader w Encode & Pad or Truncate 
train_encoded = tokenizer(list(X_train), truncation=True, padding='max_length', max_length=max_length)
test_encoded = tokenizer(list(X_test), truncation=True, padding='max_length', max_length=max_length)

class PrepDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.labels)

#Dataloader
train_data = PrepDataset(train_encoded, y_train)
test_data = PrepDataset(test_encoded, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
    
#Initialize Model 
model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

#Get Memory Usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 
#Translate from bytes to megabtyes

#Evaluate Model
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device).float()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            pred = (torch.sigmoid(logits) >= 0.5).int()
            preds.extend(pred.tolist())
            labels.extend(labels_batch.tolist())
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return accuracy, precision, recall, f1

#Train Model 
results = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device).float()
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end_time = time.time()


    #After Each Epoch, Save Data For CSV
    memory = get_memory_usage()
    accuracy, precision, recall, f1 = evaluate(model, test_loader)

    results.append({
        "epoch": epoch + 1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "loss": total_loss / len(train_loader),
        "train_time": end_time - start_time,
        "memory": memory
    })

#Save Results Into CSV for Model Comparison 
df = pd.DataFrame(results)
#Change the Name of CSV Depending on What Dataset is Being Read. 
df.to_csv("datasets/bert_dataset3.csv", index=False)