# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

#Carrega o wine dataset em wines
wines = load_wine()

#Define os parâmetros para o treinamento e teste
ncenters = 4
m = 2
numero_de_testes = 100

#Inicializações
acertos = 0

#Dividindo o dataset em treinamento e checagem
wine_train, wine_test, label_train, label_test = train_test_split(
    wines.data,wines.target,test_size=0.50,random_state=123)

#Loop para realizar o processo de treinamento e checagem X vezes

for x in range(numero_de_testes):

    #Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            wine_train.transpose(), ncenters, m, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)

    #A partir dos centros gerados testa o dataset
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            wine_test.transpose(), cntr, m, error=0.005, maxiter=1000)
    cluster_membership = np.argmax(u, axis=0)

    #Confere com a label e conclui a quantidade de acertos real do Algorítmo
    for x in range(len(cluster_membership)):
        if cluster_membership[x] == label_test[x]:
            acertos += 1

acertos /= len(cluster_membership) * numero_de_testes
    
    
    
    
    