from sklearn.metrics import mean_absolute_error, r2_score

# Calcular métricas
for model_name, model_info in results.items():
    y_pred = model_info['model'].predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f"{model_name} - MAE: {mae}, R²: {r2}")

# Exportar resultados
results_df = pd.DataFrame(results).T
results_df.to_csv('model_comparison.csv')