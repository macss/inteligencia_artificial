# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
import sklearn.cluster as skc
from sklearn import datasets as data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

dataset = data.load_iris()

# %%
#Define os parâmetros para o treinamento e teste
test_size = 0.5
ncenters = 5
n_testes = 10
f1_list = []

for _ in tqdm(range(n_testes)):
    #Dividindo o dataset em treinamento e checagem
    data_train, data_test, _, label_test = train_test_split(
                                dataset.data,dataset.target,test_size=test_size)
    
    #Implementa o Algoritmo Fuzzy C-means
    kmean_obj = skc.KMeans(n_clusters=ncenters, max_iter=10000)
    kmean_obj.fit(data_train)
    
    #A partir dos centros gerados testa o dataset
    predicted = kmean_obj.predict(data_test)
    
    f1_list.append(f1_score(label_test, predicted, average='weighted'))
    
    
f1_mean = np.mean(f1_list)
f1_error = np.std(f1_list) / np.sqrt(len(f1_list))
print('\nEm {:d} testes:'
      '\nF1 Score = {:.04f} +/- {:.04f}' \
      .format(n_testes, f1_mean, f1_error))
