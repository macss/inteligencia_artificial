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

# Carregando o dataset em iris
iris = load_iris()

# %% Definindo o como deve ser dividido o dataset
test_size = 0.3

iris_train, iris_test, label_train, label_test = train_test_split(
                                    iris.data,iris.target,test_size=test_size)

# %% Criando a função a ser minimizada pelo algoritimo genético

def f(X):
    acertos = 0
    
    #Implementa o algorítmo KNN
    neigh = KNeighborsClassifier(n_neighbors=int(X[0]),weights='uniform')
    neigh.fit(iris_train, label_train)
    
    #Prevendo novos valores
    previsao = neigh.predict(iris_test)
    
    for x in range(len(previsao)):
        if previsao[x] == label_test[x]:
            acertos += 1
                    
    #Como o algorítimo genético sempre minimiza, e querendo maximizar o numero
    #de acertos, retornamos -acertos        
    return -acertos


# %% Implementando o algorítimo genético
    
varbound = np.array([[1,20]])

model = ga(function=f,dimension=1,variable_type='int',variable_boundaries=varbound)

model.run()

# Melhor indivíduo
convergence=model.report
solution=model.output_dict