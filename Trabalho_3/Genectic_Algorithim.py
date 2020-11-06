# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:49:11 2020

@author: Marco Antonio, Samuel, Jorge

Algorítimo que tenta encontrar os melhores para Kp, Ki, e Kd, de um controlador
PID a fim de minimizar a integral de erro quadrática.
"""


# %%
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import control as ct

#Valor para o qual o sistema deve tender
TARGET_VALUE = 2;

def f(X): #F(Kp, Ki, Kd)
    #Obtendo as constantes a partir da entrada
    Kp = X[0]
    Ki = X[1]
    Kd = X[2]
    
    #Definindo as constantes do processo
    K = 2
    CT = 4
    
    s = ct.TransferFunction.s
    
    #Equação do sistema de Controle
    C = Kp + Ki/s + Kd*s
    
    #Equação do processo a ser controlado
    P = K / (CT*s + 1)
    
    #Fazendo a retroalimentação
    T = ct.TransferFunction.feedback(C*P,1)
    
    tempo, resposta = ct.step_response(T)
    
    erro = [TARGET_VALUE-resp for resp in resposta]
    
    #Cálculo da integral de erro quadratico multiplicado pelo tempo
    # print(tempo)
    # print(erro)
    itse = np.sum(tempo*np.power(erro,2))

    return itse


varbound=np.array([[0,10]]*3)
"""
print(varbound)
[[ 0 10]
 [ 0 10]
 [ 0 10]]
"""

# %%
#Variáveis reais
model=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)


model.run()

""" Melhor indivíduo(s)
convergence=model.report
solution=model.output_dict
"""

#Parametros default
#print(model.param)

"""
print(model.param)
'max_num_iteration': None,
'population_size': 100,
'mutation_probability': 0.1,
'elit_ratio': 0.01,
'crossover_probability': 0.5,
'parents_portion': 0.3,
'crossover_type': 'uniform',
'max_iteration_without_improv': None}
"""