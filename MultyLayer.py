import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
from sklearn.metrics import classification_report
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Stop words em português
stop_words = set(stopwords.words('portuguese'))

# Pré-processamento
def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Carregar dados
file_path = 'temas_e_perguntas_limpo.csv'
data = pd.read_csv(file_path)
data["Pergunta"] = data["Pergunta"].apply(preprocess_text)

# Codificar rótulos
le = LabelEncoder()
data["Tema"] = le.fit_transform(data["Tema"])

# Dividir dados em treino e teste
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["Pergunta"], data["Tema"], test_size=0.2, random_state=42, stratify=data["Tema"]
)

# Criar vocabulário
def build_vocab(texts, min_freq=1):
    tokens = list(chain(*[word_tokenize(text) for text in texts]))
    vocab = {word: idx for idx, (word, count) in enumerate(Counter(tokens).items()) if count >= min_freq}
    vocab["<pad>"] = len(vocab)  # Adicionar token de padding
    vocab["<unk>"] = len(vocab)  # Adicionar token desconhecido
    return vocab

# Texto para sequência
def text_to_sequence(text, vocab, max_len=50):
    tokens = word_tokenize(text)
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return indices[:max_len] + [vocab["<pad>"]] * (max_len - len(indices))

# Construir vocabulário
vocab = build_vocab(train_texts)

# Converter textos para sequências
train_sequences = [text_to_sequence(text, vocab) for text in train_texts]
test_sequences = [text_to_sequence(text, vocab) for text in test_texts]

# Converter para tensores PyTorch
train_sequences = torch.tensor(train_sequences, dtype=torch.long)
test_sequences = torch.tensor(test_sequences, dtype=torch.long)
train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.long)
test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.long)


# Criar DataLoader
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Modelo de Rede Neural
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 50, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Inicializar modelo
embed_dim = 100
num_classes = len(le.classes_)
model = TextClassifier(len(vocab), embed_dim, num_classes)

# Configurar otimizador e função de perda
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Avaliação
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Relatório de Classificação
print("Relatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))
