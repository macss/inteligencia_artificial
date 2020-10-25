# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:21:40 2020

@author: Renard
"""
# %%
#Pacotes utilizados
import numpy as np
import skfuzzy as skf
from skfuzzy import control as skc
from matplotlib import pyplot as plt

#Ranges
temperaturas = np.arange(0, 45, 1)
fluxos = np.arange(2,10, 0.1)

# %%
#Fuzzificação
t_ambiente = skc.Antecedent(temperaturas, 'temperatura_ambiente')
fluxo_entrada = skc.Antecedent(fluxos, 'fluxo_entrada')
potencia = skc.Consequent(np.arange(0,1, 1e-4), 'potencia_saida')

niveis = ['Baixa', 'Media', 'Alta']
t_ambiente.automf(names=niveis)
fluxo_entrada.automf(names=niveis)
potencia.automf(names=niveis)

t_ambiente.view()
fluxo_entrada.view()
potencia.view()

# %%
#Defuzzificação
regra_1 = skc.Rule(t_ambiente['Baixa'] | fluxo_entrada['Alta'], potencia['Alta'])
regra_2 = skc.Rule(t_ambiente['Media'] & fluxo_entrada['Media'], potencia['Media'])
regra_3 = skc.Rule(t_ambiente['Alta'] | fluxo_entrada['Baixa'], potencia['Baixa'])

controle = skc.ControlSystem([regra_1, regra_2, regra_3])
saida = skc.ControlSystemSimulation(controle)

# %%
#Criação de uma tabela de saída
potencia_saida = []
temp_potencia = []
for t in temperaturas:
    for f in fluxos:
        saida.input['temperatura_ambiente'] = t
        saida.input['fluxo_entrada'] = f
        saida.compute()
        temp_potencia.append(saida.output['potencia_saida'])
    temp_potencia = np.hstack(temp_potencia)
    potencia_saida.append(temp_potencia)
    temp_potencia = []
potencia_saida = np.vstack(potencia_saida)
