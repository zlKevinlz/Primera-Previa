#!/usr/bin/env python
# coding: utf-8

# In[12]:


#KEVIN ALEXANDER RODRIGUEZ BEDOYA COMPUTACION BLANDA CUBA


# In[1]:


# PROCESAMIENTO DIGITAL

# Se importan las librería numpy y las funciones de preprocesamiento

import numpy as np
from sklearn import preprocessing

# Datos de prueba
input_data = np.array([[3.1, 2.5, -1.3],
[-2.2, -3.9, -1.2],
[3.7, 0.2, -2.1],
[-9.3, 1.9, -1.5]])
print(input_data)


# In[2]:


# Binarizar los datos

data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nDatos binarizados:\n", data_binarized)


# In[3]:


# Imprimir la media y la desviación estándar

print("\nANTES:")
print(input_data)
print("Media =", input_data.mean(axis=0))
print("Desviación estándar =", input_data.std(axis=0))


# In[4]:


# Remover la media

data_scaled = preprocessing.scale(input_data)
print(input_data)
print("\nDESPUÉS:")
print("Media =", data_scaled.mean(axis=0))
print("Desviación estándar =", data_scaled.std(axis=0))


# In[5]:


# Escalamiento Min Max

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max escalamiento de datos:\n", data_scaled_minmax)


# In[6]:


# Normalización de datos

data_normalized_l1 = preprocessing.normalize(input_data,
norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data,
norm='l2')
print("\nL1 dato normalizado:\n", data_normalized_l1)
print("\nL2 dato normalizado:\n", data_normalized_l2)


# In[7]:


# Manejo de etiquetas
import numpy as np
from sklearn import preprocessing
# Se definen algunas etiquetas simples
input_labels = ['red', 'black', 'green', 'green', 'yellow', 'yellow',
'white']
# Se crea un codificador de etiquetas y se ajustan las etiquetas

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Se imprime el mapeo entre palabras y números

print("\nMapeo de etiquetas:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
    
# Codificar un conjunto de etiquetas con el codificador

test_labels = ['yellow', 'red', 'white']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Decodificar un conjunto de valores usando el codificador

encoded_values = [3, 1, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))


# In[ ]:





# In[ ]:




