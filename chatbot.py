import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from unidecode import unidecode
import spacy
from datetime import datetime

# Configuración inicial
nlp = spacy.load("es_core_news_sm")
ignore_words = {'qué', 'cómo', 'dónde', 'cuándo', 'quién'}  # Palabras a ignorar

# Carga y preprocesamiento de datos
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Preparar vocabulario
all_words = []
tags = []
documents = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = [
            token.lemma_ for token in nlp(unidecode(pattern.lower()))
            if not token.is_punct 
            and not token.is_stop
            and token.lemma_ not in ignore_words
        ]
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
    y.append(tags.index(doc[1]))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Red Neuronal 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)  # Regularización
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Configuración del modelo
input_size = len(X[0])
hidden_size = 256  # Capa oculta 
output_size = len(tags)
model = NeuralNet(input_size, hidden_size, output_size)

# Entrenamiento 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Regularización L2

# Convertir datos a tensores
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Entrenamiento con early stopping
best_loss = float('inf')
patience = 20
no_improve = 0

for epoch in range(1000):
    model.train()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

# Funciones auxiliares mejoradas
def log_unknown_question(question):
    # Registra preguntas no entendidas
    with open('unknown_questions.log', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {question}\n")

def get_suggestions():
    # Devuelve sugerencias relevantes
    priority_tags = ['tipos', 'materiales', 'precios', 'envios', 'redes']
    available = [t for t in priority_tags if t in tags]
    return available[:3] if available else random.sample(tags, min(3, len(tags)))

def predict_tag(sentence, threshold=0.65):
    # Predicción con manejo de contexto y preprocesamiento
    doc = nlp(unidecode(sentence.lower()))
    keywords = [
        token.lemma_ for token in doc 
        if not token.is_punct 
        and not token.is_stop
        and token.lemma_ not in ignore_words
    ]
    
    # Verificación de palabras clave importantes primero
    keyword_matches = {}
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = [
                token.lemma_ for token in nlp(unidecode(pattern.lower()))
                if not token.is_punct and not token.is_stop
            ]
            match_score = sum(1 for word in keywords if word in pattern_words)
            if match_score > 0:
                keyword_matches[intent['tag']] = keyword_matches.get(intent['tag'], 0) + match_score
    
    if keyword_matches:
        best_tag = max(keyword_matches.items(), key=lambda x: x[1])[0]
        return best_tag, 0.9  # Alta confianza para coincidencias claras
    
    # Si no hay coincidencia clara, usar la red neuronal
    bag = [1 if word in keywords else 0 for word in all_words]
    bag_tensor = torch.from_numpy(np.array(bag, dtype=np.float32))
    
    with torch.no_grad():
        model.eval()
        output = model(bag_tensor)
        prob = torch.softmax(output, 0)
        max_prob, idx = torch.max(prob, 0)
        return (tags[idx.item()], max_prob.item()) if max_prob > threshold else (None, None)

# Interacción con el usuario
print("\n🌸 ¡Hola! Soy el asistente de Cherry Chewy. Pregúntame sobre cualquier duda que tengas.")
print("(Escribe 'salir' cuando quieras terminar)\n")

while True:
    user_input = input("Tú: ").strip()
    if user_input.lower() in ['salir', 'exit', 'adiós']:
        print("\nBot: ¡Gracias por visitar Cherry Chewy! 💕")
        break
    
    # Manejar selección numérica
    if user_input.isdigit():
        num = int(user_input)
        suggestions = get_suggestions()
        if 1 <= num <= len(suggestions):
            user_input = suggestions[num-1]
        else:
            print("\n⚠️ Por favor elige un número entre 1 y", len(suggestions))
            continue
    
    tag, confidence = predict_tag(user_input)
    
    if not tag:
        log_unknown_question(user_input)
        print("\nBot: No estoy segura de entender. ¿Te refieres a algo de esto?")
        suggestions = get_suggestions()
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion.capitalize()}")
        print("\nResponde con el número o reformula tu pregunta ❓")
    else:
        # Buscar la respuesta correspondiente
        response = next(
            (random.choice(intent['responses']) 
             for intent in intents['intents'] 
             if intent['tag'] == tag),
            "No tengo información sobre eso."
        )
        print(f"\nBot: {response}")