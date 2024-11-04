import pandas as pd

# Cargar el conjunto de datos
data = pd.read_csv('data.csv')

# Mostrar las primeras filas del conjunto de datos
print(data.head())

# Verificar si hay valores faltantes
missing_values = data.isnull().sum()
print("Valores faltantes:\n", missing_values)

# Visualizar la distribución del precio
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(data['precio'], bins=30, color='pink', edgecolor='black')
plt.title('Distribución de Precio')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()
