import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from unidecode import unidecode
import spacy

# Cargar modelo de lenguaje spaCy
nlp = spacy.load("es_core_news_sm")

# --- Preprocesamiento de datos (igual que antes) ---
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
documents = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = [token.lemma_ for token in nlp(unidecode(pattern.lower())) if not token.is_punct and not token.is_stop]
        all_words.extend(words)
        documents.append((words, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Crear datos de entrenamiento
X = []
y = []
for doc in documents:
    bag = [1 if word in doc[0] else 0 for word in all_words]
    X.append(bag)
    label = tags.index(doc[1])
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# --- Red Neuronal con PyTorch ---
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = len(X[0])
hidden_size = 128
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

for epoch in range(1000):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Época {epoch+1}, Pérdida: {loss.item():.4f}')

# --- Función para predecir ---
def predict(sentence):
    words = [token.lemma_ for token in nlp(unidecode(sentence.lower())) if not token.is_punct and not token.is_stop]
    bag = [1 if word in words else 0 for word in all_words]
    bag_tensor = torch.from_numpy(np.array(bag, dtype=np.float32))
    output = model(bag_tensor)
    _, predicted = torch.max(output, 0)
    return tags[predicted.item()]

# --- Interacción con el usuario ---
print("¡Chatbot con red neuronal activado! Escribe 'salir' para terminar.")
while True:
    user_input = input("Tú: ")
    if user_input.lower() == 'salir':
        break
    predicted_tag = predict(user_input)
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            print("Bot:", random.choice(intent['responses']))
            break