import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import yaml

# Cargar parámetros desde params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Evaluar modelos
def evaluate_models(X_test, y_test):
    results = {}

    for model_name in params['models']:
        model = joblib.load(f'models/{model_name}.joblib')
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[model_name] = mse

    return results

if __name__ == "__main__":

    results = evaluate_models(X_test, y_test)
    print(results)
