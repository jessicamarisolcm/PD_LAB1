import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import yaml

# Cargar parámetros desde params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Cargar el dataset
def load_data(filepath):
    data = pd.read_csv(data.csv)
    return data

# Preprocesar datos
def preprocess_data(data):
    # Identificar columnas numéricas y categóricas
    numeric_features = params['preprocessing']['numeric_features']
    categorical_features = params['preprocessing']['categorical_features']

    # Transformadores
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    # Combinación de transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Aplicar transformaciones
    X = data.drop('price', axis=1)  
    y = data['price']  
    # Dividir el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['data']['test_size'], random_state=42)

    # Ajustar y transformar los datos de entrenamiento
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    data = load_data('data/dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data)
