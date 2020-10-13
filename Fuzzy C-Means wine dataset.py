# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
import pandas as pd
import skfuzzy as fuzz
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
ncenters = 5
numero_de_testes = 100
m_base = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 10]

#Inicialização dos resultados
acertos_total = []
iteracoes_total = []

for m in m_base:
    #Inicializações
    acertos = 0
    iteracoes = []

    #Loop para realizar o processo de treinamento e checagem X vezes
    for _ in range(numero_de_testes):
        #Dividindo o dataset em treinamento e checagem
        wine_train, wine_test, _, label_test = train_test_split(
                                wines.data,wines.target,test_size=test_size)

        #Implementa o Algoritmo Fuzzy C-means
        cntr, _, _, _, _, n_iter, _ = fuzz.cluster.cmeans(
                wine_train.transpose(), ncenters, m, error=0.005, maxiter=10000, init=None)
        iteracoes.append(n_iter)

        #A partir dos centros gerados testa o dataset
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                wine_test.transpose(), cntr, m, error=0.005, maxiter=10000)
        cluster_membership = np.argmax(u, axis=0)

        #Confere com a label e conclui a quantidade de acertos real do Algorítmo
        for x in range(len(cluster_membership)):
            if cluster_membership[x] == label_test[x]:
                acertos += 1

    acertos /= len(cluster_membership) * numero_de_testes
    
    acertos_total.append(np.round(acertos,4))
    iteracoes_total.append(np.round(np.mean(iteracoes),0))
    
    print('Acertos com m =', str(m), 'e', str(np.round(np.mean(iteracoes),0)),
          'iterações médias:', acertos*100, '%')
