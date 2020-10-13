# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
import pandas as pd
import sklearn.cluster as skc
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

#Carrega o wine dataset em wines
wines = load_wine()

# %%

dataset = pd.DataFrame(np.column_stack((wines.data, wines.target)),
                   columns = wines.feature_names + ['Target'])
print(dataset)

# %%
#Define os parâmetros para o treinamento e teste
test_size = 0.5
ncenters = 3
numero_de_testes = 100

#Inicializações
acertos = 0

#Loop para realização das simulações
for _ in range(numero_de_testes):
    #Dividindo o dataset em treinamento e checagem
    wine_train, wine_test, _, label_test = train_test_split(
                                wines.data,wines.target,test_size=test_size)

    #Implementa o Algoritmo Fuzzy C-means
    kmean_obj = skc.KMeans(n_clusters=ncenters, max_iter=10000)
    kmean_obj.fit(wine_train)

    # #A partir dos centros gerados testa o dataset
    cluster_membership = kmean_obj.predict(wine_test)

    #Confere com a label e conclui a quantidade de acertos real do Algorítmo
    for x in range(len(cluster_membership)):
        if cluster_membership[x] == label_test[x]:
            acertos += 1

acertos /= len(cluster_membership) * numero_de_testes

print('Acertos:', acertos*100, '%')
