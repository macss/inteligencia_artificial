# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 08:30:14 2020

@author: MarcoAntonio, Samuel, Jorge
"""


# %%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# Carregando o dataset em iris
iris = load_iris()

# %% Definindo o como deve ser dividido o dataset
test_size = 0.5

iris_train, iris_test, label_train, label_test = train_test_split(
                                    iris.data,iris.target,test_size=test_size)

# %% Criando a função a ser minimizada pelo algoritimo genético

def f(X):
    global iris_train, label_train, iris_test, label_test
    
    #Implementa o algorítmo KNN
    neigh = KNeighborsClassifier(n_neighbors=int(X[0]),weights='uniform')
    neigh.fit(iris_train, label_train)
    
    #Prevendo novos valores
    previsao = neigh.predict(iris_test)
    
    f1 = f1_score(label_test, previsao, average='weighted')

    return -f1


# %% Implementando o algorítimo genético
    
varbound = np.array([[1,20]])

model = ga(function=f,
           dimension=1,
           variable_type='int',
           variable_boundaries=varbound)

model.run()

# Melhor indivíduo
convergence=model.report
solution=model.output_dict

# %%
#Testando o modelo com dados aleatorizados para obter o F1 Score médio
n_testes = 10

f1_list = []
for _ in tqdm(range(n_testes)):
    _, iris_test, _, label_test = train_test_split(
                                    iris.data, iris.target, test_size=0.5)
    f1_list.append((-1)* f(solution['variable']))


f1_mean = np.mean(f1_list)
f1_error = np.std(f1_list) / np.sqrt(len(f1_list))
print('\nEm {:d} testes:'
      '\nF1 Score = {:.04f} +/- {:.04f}' \
      .format(n_testes, f1_mean, f1_error))
