import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
import nltk

# Baixar stop words em português
nltk.download('stopwords')
portuguese_stop_words = stopwords.words('portuguese')

# Função de limpeza de texto
def preprocess_text(text):
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'\d+', '', text)  # Remove números
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # Remove espaços extras
    return text

# Carregar os dados
file_path = 'temas_e_perguntas_limpo.csv'
data = pd.read_csv(file_path)

# Limpar o texto das perguntas
data["Pergunta"] = data["Pergunta"].apply(preprocess_text)

# Codificar os rótulos (Temas)
le = LabelEncoder()

# Configurar validação cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Métricas para avaliação
metrics = {"precision": [], "recall": [], "f1_score": [], "accuracy": []}

# Inicializar o TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=portuguese_stop_words, max_features=5000)

# Validação cruzada
for fold, (train_index, test_index) in enumerate(skf.split(data["Pergunta"], data["Tema"])):
    print(f"=== Fold {fold + 1} ===")
    
    # Dividir dados em treino e teste
    train_df = data.iloc[train_index]
    test_df = data.iloc[test_index]

    # Transformar perguntas em uma matriz TF-IDF
    X_train = vectorizer.fit_transform(train_df["Pergunta"])
    X_test = vectorizer.transform(test_df["Pergunta"])

    # Codificar os rótulos (temas)
    y_train = le.fit_transform(train_df["Tema"])
    y_test = le.transform(test_df["Tema"])

    # Treinar o modelo: Random Forest com hiperparâmetros ajustados
    classifier = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    classifier.fit(X_train, y_train)

    # Previsões
    y_pred = classifier.predict(X_test)

    # Avaliar as métricas
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # Salvar as métricas do fold
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1_score"].append(f1)
    metrics["accuracy"].append(accuracy)

    # Exibir o relatório de classificação
    print("Relatório de Classificação para este fold:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print("\n")

# Resultados finais
print("=== Resultados Médios de Validação Cruzada ===")
for metric, scores in metrics.items():
    print(f"{metric.capitalize()}: {np.mean(scores):.2f}")
