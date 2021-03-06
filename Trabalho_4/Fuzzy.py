# -*- coding: utf-8 -*-
# %%
#Pacotes utilizados
import numpy as np
import pandas as pd
import skfuzzy as skf

from skfuzzy import control as skc
from sklearn import datasets as data
from sklearn import model_selection as skms
# from matplotlib import pyplot as plt
from beecolpy import abc
from sklearn.metrics import f1_score
from tqdm import tqdm
import time

aplicar_otimização = False #Atenção, vai demorar MUITO se "True"

#Esta função aplica o modelo fuzzy de classificação com funções de pertinência
#triangulares, com vértices localizados em [model].
def apply_model(data, label, model, plot_model: bool=False):
    #Regras
    #A variável "sepal_width" foi retirada do modelo após efetuado teste de correlação
    #explicado e executado no segundo bloco do código.
    sepal_length = skc.Antecedent(np.arange(0.5, 10, 0.1), 'sepal_length')
    petal_length = skc.Antecedent(np.arange(0.5, 10, 0.1), 'petal_length')
    petal_width = skc.Antecedent(np.arange(0.1, 5.0, 0.1), 'petal_width')
    group = skc.Consequent(np.arange(0, 2, 1e-2), 'Group')
    
    #Função de aplicação do modelo fuzzy
    #Os vértices inicial e central das funções de "pequena" e "G0" são fixos em pertinência
    #máxima no início da escala, enquanto os de funções "grande" e "G2" no final da escala.
    #Dessa forma a quantidade de graus de liberdade para cada entrada diminuiu de 9 para 5.
    sepal_length['Pequena'] = skf.trimf(sepal_length.universe, [0.5, 0.5, model[0]])    #<
    sepal_length['Media'] = skf.trimf(sepal_length.universe, [model[1], model[2], model[3]])
    sepal_length['Grande'] = skf.trimf(sepal_length.universe, [model[4], 10, 10])
    
    petal_length['Pequena'] = skf.trimf(petal_length.universe, [0.5, 0.5, model[5]])
    petal_length['Media'] = skf.trimf(petal_length.universe, [model[6], model[7], model[8]])
    petal_length['Grande'] = skf.trimf(petal_length.universe, [model[9], 10, 10])
    
    petal_width['Pequena'] = skf.trimf(petal_width.universe, [0.1, 0.1, model[10]])
    petal_width['Media'] = skf.trimf(petal_width.universe, [model[11], model[12], model[13]])
    petal_width['Grande'] = skf.trimf(petal_width.universe, [model[14], 5, 5])
    
    group['G0'] = skf.trimf(group.universe, [0, 0, model[15]])
    group['G1'] = skf.trimf(group.universe, [model[16], model[17], model[18]])
    group['G2'] = skf.trimf(group.universe, [model[19], 2, 2])
    
    #As regras foram definidas analizando-se uma parte do conjunto de teste
    #em tabela, onde foram determinados os valores mínimo, médio e máximo de
    #cada variável de entrada e relacionada a posição (próximo de min = pequena,
    #próxima da média = media e próxima da máxima = grande) com o grupo de saída.
    regra_1 = skc.Rule(sepal_length['Media'] & petal_length['Pequena'] &
                       petal_width['Pequena'], group['G0'])
    
    regra_2 = skc.Rule(sepal_length['Media'] & petal_length['Media'] &
                       petal_width['Media'], group['G1'])
    
    regra_3 = skc.Rule(sepal_length['Grande'] & petal_length['Grande'] &
                       petal_width['Grande'], group['G2'])
    
    #Montado o modelo
    controle = skc.ControlSystem([regra_1, regra_2, regra_3])
    saida = skc.ControlSystemSimulation(controle)
    
    #Plota os gráficos no momento da validação
    if plot_model:
        sepal_length.view()
        petal_length.view()
        petal_width.view()
        group.view()
    
    predicted = []
    for i in range(len(data)):
        saida.input['sepal_length'] = data[i][0]
        saida.input['petal_length'] = data[i][2]
        saida.input['petal_width'] = data[i][3]
        saida.compute()
        
        #Para definir o grupo, é efetuado um arredondamento da saída do modelo
        #fuzzy com 0 casas decimais a fim de estabelecer o grupo de pertencimento
        #entre 0, 1 ou 2.
        #Dessa forma:
            #       saida < 0.5  --> Grupo 0
            # 0.5 < saida < 1.5  --> Grupo 1
            # 1.5 < saida        --> Grupo 2
        predicted.append(np.round(saida.output['Group']))
    
    return f1_score(label, predicted, average='weighted')

#Carrega o dataset e separa em conjunto de treino e validação
iris = data.load_iris()
dataset = np.column_stack((iris.data, iris.target))
dataset_columns = iris.feature_names + ['Group']

#Definida seed (random_state) para a randomização dos elementos para facilitar
#a reprodutibilidade, porém, o algoritmo funciona normalmente se este parâmetro
#for retirado.
iris_train, iris_validate, label_train, label_validate = skms.train_test_split(
    iris.data, iris.target, test_size=0.5, random_state=1)

# %%
#Teste de correlação das características com os grupos
covariance = []
for i in range(len(iris_train[0])):
    temp_cov = np.cov(iris_train[:,i], label_train)[0,1]/ \
                        (np.std(iris_train[:,i]) * np.std(label_train))
    covariance.append(temp_cov)
print('\nCovariância normalizada de cada parâmetro com a classificação:\n',
      pd.DataFrame([covariance], columns=iris.feature_names),'\n')

#A correlação da "sepal width" é inversamente proporcional à classificação e seu
#valor é consideravelmente baixo (abs<0.5), sendo assim, esta variável será suprimida
#para reduzir os graus de liberdade necessários para o algoritmo de otimização
#resolver.

#As demais características são relevantes pois possuem covariância normalizada
#muito próximas de 1.

# %%
if aplicar_otimização:
    def cost_function(x):
        global iris_train, label_train
        
        try:
            #Verificação da validade dos valores testados.
            #Quando os vértices das funções de pertinência assumem posições
            #inválidas, o algoritmo para a execução e retorna NaN antes de prosseguir,
            #diminuindo assim o custo computacional da simulação.
            #sepal_length
            if (x[0] <= x[1]):
                return np.nan
            
            for i in range(2,4):
                if x[i-1] > x[i]:
                    return np.nan
                
            if (x[3] <= x[4]):
                return np.nan
    
            #petal_length
            if (x[5] <= x[6]):
                return np.nan
        
            for i in range(7,9):
                if x[i-1] > x[i]:
                    return np.nan
            
            if (x[8] <= x[9]):
                return np.nan
    
            #petal_width
            if (x[10] <= x[11]):
                return np.nan
            
            for i in range(12,14):
                if x[i-1] > x[i]:
                    return np.nan
            
            if (x[13] <= x[14]):
                return np.nan
            
            #group
            if (x[15] <= x[16]):
                return np.nan
            
            for i in range(17,19):
                if x[i-1] > x[i]:
                    return np.nan
                
            if (x[18] <= x[19]):
                return np.nan
            
            #Caso os valores do modelo estejam dentro do tolerado, aplica
            #o modelo no conjunto de dados de treinamento.
            return apply_model(iris_train, label_train, x) * (-1)
        
        except:
            #Sim, as vezes os valores do modelo geram erro no pacote skfuzzy.
            #Mesmo com os valores aparentemente dentro da faixa válida, algumas vezes
            #o modelo Fuzzy diverge. Como algoritmos de meta-heurística testam muitos
            #valores do domínio da função custo, é muito raro deste evento não ocorrer.
            #Dessa forma a função custo retorna NaN caso ocorra um erro no modelo
            #fuzzy testado.
            return np.nan
    
    #Limites do espaço de busca
    sep_le = [(0.5, 10) for _ in range(5)]
    pet_le = [(0.5, 10) for _ in range(5)]
    pet_wi = [(0.1, 5) for _ in range(5)]
    g = [(0, 2) for _ in range(5)]
    boundaries = sep_le + pet_le + pet_wi + g
    
    #Usamos a colonia de abelhas artificial por ser um algoritmo mais rápido neste
    #tipo de problema, e, já possui proteção contra NaNs, aumentando a probabilidade
    #de convergência, contudo, qualquer algoritmo de otimização numérica
    #capaz de avaliar sistemas multi-dimensionais com eficiência (nesse caso no R^20)
    #pode ser aplicado aqui.
    #Caso seja utilizado um algoritmo que não esteja preparado para lidar com NaNs, pode-se
    #trocar o valor de retorno da função custo para Inf, contudo, no algoritmo
    #utilizado neste trabalho, a utilização de NaN aumenta a convergência do algoritmo.
    
    #Criação do objeto e das variáveis de armazenamento do histórico.
    abc_obj = abc(cost_function, boundaries, scouts=0.1, iterations=1, nan_protection=True)
    total_iterations = 0 #Histórico do total de iterações executadas
    cost_history = [] #Historico do avanço da função custo

# %%
if aplicar_otimização:
    #Executa a otimização
    cost = cost_function(abc_obj.get_solution())
    min_cost = 0.9 #valor objetivo a ser atingido no conjuunto de treinamento
    counter_max = 1000000 #Executa até 1.000.000 iterações máximas (para se convergir antes)
    counter = 0 #valor inicial do contador
    while (np.isnan(cost) or (abs(cost) < min_cost)) and (counter < counter_max):
        abc_obj.fit()
        cost = cost_function(abc_obj.get_solution())
        cost_history.append(abs(cost))
        total_iterations += abc_obj.get_status()[0]
        counter += 1
    
    print('\nAssertividade do treinamento:' + str(cost_function(abc_obj.get_solution())*100) + '%\n')
    model = abc_obj.get_solution()

else:
    #Modelo encontrado pela meta-heuristica (salvo para evitar acidentes, com
    #assertividade acima de 93% no conjunto de validação)
    model = [7.820832190853921, 2.31482724609497, 5.6662017396228395,
              6.671917724767254, 2.0239317860615778, 4.40492983366766,
              2.018972446124932, 6.332068172496837, 8.53634119822423,
              1.9273473221936863, 1.3584252936962893, 0.9409301064887465,
              2.054738187216678, 4.806052425602064, 0.10647104158758393,
              1.1433952151944697, 1.0483687468076053, 1.6248452828555107,
              1.7368569274222652, 0.8436899666614793]

# %%
#Validação
f1 = apply_model(iris_validate, label_validate, model, plot_model=True)
time.sleep(1)

print('\nF1 Score obtido no modelo: {:.04f}\n'.format(f1))
time.sleep(1)

# %%
#Testando o modelo com dados aleatorizados para obter o F1 Score médio
n_testes = 1000

f1_list = []
for _ in tqdm(range(n_testes)):
    _, data_val, _, label_val = skms.train_test_split(
                                    iris.data, iris.target, test_size=0.5)
    f1_list.append(apply_model(data_val, label_val, model))


f1_mean = np.mean(f1_list)
f1_error = np.std(f1_list) / np.sqrt(len(f1_list))
print('\nEm {:d} testes:'
      '\nF1 Score = {:.04f} +/- {:.04f}' \
      .format(n_testes, f1_mean, f1_error))
