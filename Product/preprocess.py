import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cargar el conjunto de datos
data = pd.read_csv('data.csv')

# Preprocesamiento
X = data.drop('precio', axis=1)
y = data['precio']

# Dividir en conjuntos de entrenamiento y prueba
train_size = 0.8  # Se puede cambiar a partir de params.yaml
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

# Codificación y escalado
numeric_features = ['area', 'num_rooms', 'num_bathrooms']  # Se puede cargar desde params.yaml
categorical_features = ['location', 'type']  # Se puede cargar desde params.yaml

# Normalización
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# OneHot Encoding
encoder = OneHotEncoder(sparse=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Unir las características codificadas y numéricas
import numpy as np
X_train_final = np.hstack([X_train_encoded, X_train[numeric_features].values])
X_test_final = np.hstack([X_test_encoded, X_test[numeric_features].values])
