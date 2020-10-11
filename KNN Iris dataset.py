# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
 
#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target 
#iris.target
#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=5,weights="uniform")
neigh.fit(X, y)
#Prevendo novos valores
#print(neigh.predict([[5.9, 3. , 5.1, 1.8]]))
