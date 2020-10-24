# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#Carrega o wine dataset em wines
wines = load_wine()

# %%

dataset = pd.DataFrame(np.column_stack((wines.data, wines.target)),
                   columns = wines.feature_names + ['Target'])
print(dataset)

# %%
test_size = 0.5
vizinhos = 10
numero_de_testes = 100
pesos = ['uniform', 'distance']

#Inicialização dos resultados
acertos_total = []

for peso in pesos:
    for numero_de_vizinhos in range(1, vizinhos+1):
        #Inicialização
        acertos = 0

        #Loop para realização das simulações
        for _ in range(numero_de_testes):
            #Dividindo o dataset em treinamento e checagem
            wine_train, wine_test, label_train, label_test = train_test_split(
                                    wines.data,wines.target,test_size=test_size)

            #Implementa o Algoritmo KNN
            neigh = KNeighborsClassifier(n_neighbors=numero_de_vizinhos,weights=peso)
            neigh.fit(wine_train, label_train)

            #Prevendo novos valores
            previsao = neigh.predict(wine_test)

            for x in range(len(previsao)):
                if previsao[x] == label_test[x]:
                    acertos += 1

        acertos /= len(previsao) * numero_de_testes

        acertos_total.append(np.round(acertos,4))
        print('Acertos com método de peso', peso, 'e', str(numero_de_vizinhos),
              'vizinhos:', acertos*100, '%')
