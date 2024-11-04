import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import yaml

# Cargar parámetros desde params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Entrenar modelos
def train_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        results[model_name] = {'model': model, 'mse': mse}
        # Guardar el modelo entrenado
        joblib.dump(model, f'models/{model_name}.joblib')

    return results

if __name__ == "__main__":
    # Aquí debes cargar los datos preprocesados como se hizo anteriormente
    # X_train, y_train = ... (cargar desde el archivo o directamente)
    results = train_models(X_train, y_train)
    print(results)
