# -*- coding: utf-8 -*-
# %%
#Pacotes utilizados
import numpy as np
import skfuzzy as skf
from skfuzzy import control as skc
from matplotlib import pyplot as plt

#Ranges
#Temperatura ambiente
temperaturas = np.arange(0, 45, 1)

#Chuveiros elétricos (não ducha) operam com fluxo variando entre 4 e 6 l/m
fluxos = np.arange(3,7, 0.1)

# %%
#Regras de Fuzzificação
t_ambiente = skc.Antecedent(temperaturas, 'temperatura_ambiente')
fluxo_entrada = skc.Antecedent(fluxos, 'fluxo_entrada')
potencia = skc.Consequent(np.arange(0,1, 1e-4), 'potencia_saida')

niveis = ['Baixa', 'Media', 'Alta']
#Temperatura ambiente possui funções de pertinência simples, portanto foi
#gerado automaticamente.
t_ambiente.automf(names=niveis)

#Fluxos de entrada inferiores a 4l/m são obrigatoriamente baixos.
#Fluxos de entrada superiores a 6l/m são obrigatoriamente altos.
#Portanto foi utilizada funções de pertinência trapezoidais, garantindo
#a pertinência adequada nos fluxos abaixo e acima do "limite".
fluxo_entrada['Baixa'] = skf.trapmf(fluxo_entrada.universe, [3,3,4,6])
fluxo_entrada['Media'] = skf.trapmf(fluxo_entrada.universe, [4,5,5,6])
fluxo_entrada['Alta'] = skf.trapmf(fluxo_entrada.universe, [4,6,7,7])

#Ajustada a função de pertinência da potência para que valores muito baixos ou
#altos de potência não gerem valores de pertinência na faixa média de potência.
potencia['Baixa'] = skf.trimf(potencia.universe, [0,0,0.33])
potencia['Media'] = skf.trimf(potencia.universe, [0.2,0.5,0.8])
potencia['Alta'] = skf.trimf(potencia.universe, [0.66,1,1])

#Visualização das funções de pertinência.
t_ambiente.view()
fluxo_entrada.view()
potencia.view()

# %%
#Regras de Defuzzificação
regra_1 = skc.Rule(t_ambiente['Baixa'] | fluxo_entrada['Alta'], potencia['Alta'])
regra_2 = skc.Rule(t_ambiente['Media'] & fluxo_entrada['Media'], potencia['Media'])
regra_3 = skc.Rule(t_ambiente['Alta'] | fluxo_entrada['Baixa'], potencia['Baixa'])

controle = skc.ControlSystem([regra_1, regra_2, regra_3])
saida = skc.ControlSystemSimulation(controle)

# %%
#Criação de uma tabela de saída (para implementar no controlador)
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

# %%
#Plot de exemplos

#Potência percentual de saída por temperatura, com fluxo constante.
plt.figure()
plt.plot(temperaturas,potencia_saida[:,10], label='Fluxo = $4l/m$')
plt.plot(temperaturas,potencia_saida[:,20], label='Fluxo = $5l/m$')
plt.plot(temperaturas,potencia_saida[:,30], label='Fluxo = $6l/m$')
plt.legend(loc='best')
plt.grid(True)

#Potência percentual de saída por fluxo com temperatura constante.
plt.figure()
plt.plot(fluxos,potencia_saida[5,:], label='Temperatura ambiente = $5$ \u00b0C')
plt.plot(fluxos,potencia_saida[25,:], label='Temperatura ambiente = $25$ \u00b0C')
plt.plot(fluxos,potencia_saida[35,:], label='Temperatura ambiente = $35$ \u00b0C')
plt.legend(loc='best')
plt.grid(True)
