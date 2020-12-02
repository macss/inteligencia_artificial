# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:12:28 2020

@author: Renard
"""
import math as mt
import numpy as np
from sklearn import datasets as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from beecolpy import bin_abc
from geneticalgorithm import geneticalgorithm as ga
from tqdm import tqdm

#Valores: 'ga' ou 'abc': aplica otimização com o método desejado (GA é demorado!)
#         Qualquer outro valor: Aplica modelo já obtido anteriormente.
otimizar = 'abc'

dataset = data.load_iris()

train_data, test_data, train_labels, test_labels = train_test_split(
                                dataset.data, dataset.target,
                                test_size=0.5, random_state=1)

# %%
#Otimização para achar a melhor função de ativação e
#número de neurônios na camada oculta
def bin2int(bit):
    return int(np.sum([(bit[::-1][i])*mt.pow(2,i) for i in range(len(bit))]))


def translate_hl_size(code):
    hl_len = bin2int(code) + 1
    return (hl_len,)


def translate_activation_function(code):
    function_code = bin2int(code)
    if function_code == 0:
        act_func = 'identity'
    elif function_code == 1:
        act_func = 'logistic'
    elif function_code == 2:
        act_func = 'tanh'
    elif function_code == 3:
        act_func = 'relu'
    else:
        raise Exception()
    return act_func


def cost_function(x):
    global train_data, test_data, train_labels, test_labels
    
    try:
        ann_obj = MLPClassifier(hidden_layer_sizes = translate_hl_size(x[:-2]),
                                activation = translate_activation_function(x[-2:]),
                                learning_rate = 'adaptive',
                                alpha = 0.1,
                                max_iter = 5000)
        
        ann_obj.fit(train_data, train_labels)
        
        predicted = ann_obj.predict(test_data)
        
        return (-1) * f1_score(train_labels, predicted, average='weighted')
    
    except:
        return np.inf


# %%
#Aplica o algoritmo de otimização
if (otimizar in ['ga', 'abc']):
    if (otimizar == 'ga'):
        ga_obj = ga(function = cost_function,
                    dimension = 6,
                    variable_type = 'bool')
    
        ga_obj.run()
    
        neuronios = translate_hl_size(ga_obj.best_variable[:-2])
        funcao_ativacao = translate_activation_function(ga_obj.best_variable[-2:])

    elif (otimizar == 'abc'):
        n_iter = 100
        
        abc_obj = bin_abc(function = cost_function,
                          bits_count = 6,
                          colony_size = 20,
                          scouts = 0.1,
                          iterations = 1)
        
        for _ in tqdm(range(n_iter)):
            abc_obj.fit()
            
        neuronios = translate_hl_size(abc_obj.get_solution()[:-2])
        funcao_ativacao = translate_activation_function(abc_obj.get_solution()[-2:])
    
else:
    #Melhor modelo encontrado
    
    #Obtidos com ABC (iterações: 10, scouts: 0.1, tamanho de colônia: 20)
    # neuronios = (2,)
    # funcao_ativacao = 'relu'
    
    #Obtidos com GA (iterações: 36 (max: None), população: 100)
    neuronios = (1,)
    funcao_ativacao = 'identity'


# %%
#Validação
n_testes = 1000

ann_obj = MLPClassifier(hidden_layer_sizes = neuronios,
                        activation = funcao_ativacao,
                        learning_rate = 'adaptive',
                        alpha = 0.1,
                        max_iter = 5000)

ann_obj.fit(train_data, train_labels)

f1_list = []
for _ in tqdm(range(n_testes)):
    _, val_data, _, val_labels = train_test_split(
                                                dataset.data, dataset.target,
                                                test_size=0.5)

    predicted = ann_obj.predict(val_data)

    f1_list.append(f1_score(val_labels, predicted, average='weighted'))


f1_mean = np.mean(f1_list)
f1_error = np.std(f1_list) / np.sqrt(len(f1_list))
print('\nEm {:d} testes:'
      '\nF1 Score = {:.04f} +/- {:.04f}' \
      .format(n_testes, f1_mean, f1_error))

