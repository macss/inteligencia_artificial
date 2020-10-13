# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: Marco Antonio, Samuel e Jorge
"""

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
 
#Carrega o wine dataset em wines 
wines = load_wine()
vizinhos = 10

# %%

dataset = pd.DataFrame(np.column_stack((wines.data, wines.target)),
                   columns = wines.feature_names + ['Target'])
print(dataset)

# %%
for i in range(1, vizinhos+1):
    #Inicialização
    acertos = 0
    numero_de_testes = 100
    numero_de_vizinhos = i
    
    #Dividindo o dataset em treinamento e checagem
    wine_train, wine_test, label_train, label_test = train_test_split(
        wines.data,wines.target,test_size=0.5,random_state=123)
    
    #Loop para realização das simulações
    for x in range(numero_de_testes):
        
        #Implementa o Algoritmo KNN
        neigh = KNeighborsClassifier(n_neighbors=numero_de_vizinhos,weights="distance")
        neigh.fit(wine_train, label_train)
        
        #Prevendo novos valores
        previsao = neigh.predict(wine_test)
        
        for x in range(len(previsao)):
            if previsao[x] == label_test[x]:
                acertos += 1
                
    acertos /= len(previsao) * numero_de_testes
    
    print('acertos com ', str(i), 'vizinhos: ', acertos*100, '%')
