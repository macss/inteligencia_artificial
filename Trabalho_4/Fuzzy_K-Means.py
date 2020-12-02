# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
import skfuzzy as fuzz
from sklearn import datasets as data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

dataset = data.load_iris()

# %%
#Define os parâmetros para o treinamento e teste
test_size = 0.5
ncenters = 3
n_testes = 1000
m_base = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 10]

#Inicialização dos resultados
f1_lists = []

for m in tqdm(m_base):
    f1 = []

    #Loop para realizar o processo de treinamento e checagem X vezes
    for _ in range(n_testes):
        #Dividindo o dataset em treinamento e checagem
        data_train, data_test, _, label_test = train_test_split(
                            dataset.data, dataset.target, test_size=test_size)

        #Implementa o Algoritmo Fuzzy C-means
        cntr, _, _, _, _, n_iter, _ = fuzz.cluster.cmeans(
                            data_train.transpose(), ncenters, m, error=0.005, 
                            maxiter=10000, init=None)

        #A partir dos centros gerados testa o dataset
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                data_test.transpose(), cntr, m, error=0.005, maxiter=10000)
        predicted = np.argmax(u, axis=0)

        f1.append(f1_score(label_test, predicted, average='weighted'))

    f1_lists.append(f1)

f1_mean_list = [np.mean(f1) for f1 in f1_lists]

f1_index = np.nanargmax(f1_mean_list)
f1_mean = f1_mean_list[f1_index]
f1_error = np.std(f1_lists[f1_index]) / np.sqrt(len(f1_lists[f1_index]))

print('\nEm {:d} testes, com m = {:.01f}:' \
      '\nF1 Score = {:.04f} +/- {:.04f}' \
      .format(n_testes, m_base[f1_index], f1_mean, f1_error))
