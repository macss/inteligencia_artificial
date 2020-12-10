# -*- coding: utf-8 -*-

# %%
#Pacotes e dados iniciais
import numpy as np
import math as mt
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from covariance import correlation

# Altera os parâmetros de geração de gráficos para o padrão de publicação.
publicar_graficos = False

# Carrega os dados.
# Inicialmente estes dados são referentes à parcela de treinamento e teste.
load = np.loadtxt('buck_id.dat')
u = load[:, 1]
y = load[:, 2]

if publicar_graficos:
    plt.rc('font', size=42)
    plt.rc('lines', linewidth=8, markersize=15)
else: #Default
    plt.rc('font', size=10)
    plt.rc('lines', linewidth=1.5, markersize=6)


# %%
# Determinando a decimação

# Como não sabemos se os sinais foram amostrados com frequência alta de mais 
# (o que causa problemas já que passamos a modelar ruído além do sinal), 
# decide-se o valor no qual os dados devem ser decimados a fim de ajustar 
# essa frequência de amostragem.

# A decisão deste valor é dada utilizando-se a autocorrelação do sinal de 
# saída (y), onde o ponto onde a autocorrelação atinge a primeira inflexão 
# (mudança de sinal) tem seu valor dividido por 20 (tirado do livro do 
# Aguirre) para determinar o valor da decimação.

# A análise de autocorrelação é feita sobre o sinal de saída para verificar a 
# autocorrelação linear e sobre o sinal de saída ao quadrado para verificar a 
# autocorrelação não-linear.

# A decimação indica de quantos em quantos pontos serão tomados,
# por exemplo, caso a decimação seja 5, do conjunto de dados inicial, será 
# considerado um em cada 5 pontos.

ry, ty, _, _ = correlation(y)
ry2, ty2, _, _ = correlation(np.power(y,2))

ty_min = np.where(np.diff(ry)>=0)[0][0]
ty2_min = np.where(np.diff(ry)>=0)[0][0]

plt.figure()
plt.plot(ty, ry, 'b-', label='$r_{yy}$', zorder=2)
plt.scatter(ty[ty_min], ry[ty_min], color='red', zorder=3)
plt.legend()
plt.grid(True, zorder=0)

plt.figure()
plt.plot(ty2, ry2, 'b-', label='$r_{{y^2}{y^2}}$', zorder=2)
plt.scatter(ty2[ty2_min], ry2[ty2_min], color='red', zorder=3)
plt.legend()
plt.grid(True, zorder=1)

tau_y = ty[ty_min] / 20
tau_y2 = ty2[ty2_min] / 20

decimacao = int(min([tau_y, tau_y2]))
print('Os dados podem ser decimados em {:d} para ' \
      'evitar os efeitos de super amostragem.'.format(decimacao))

# Decima os dados de treinamento e teste
y = y[::decimacao]
u = u[::decimacao]

# Carrega e já decima os dados de validação
load = np.loadtxt('buck_val.dat')
u_val = load[::decimacao, 1]
y_val = load[::decimacao, 2]


# %%
# Análise de variância

# Mesmo não sendo uma forma definitiva de definir o atraso real do modelo, 
# pode-se utilizar a correlação cruzada entre a entrada (u) e a saída (y) 
# para estimar o atraso máximo do modelo.

ruy, tuy, _, _ = correlation(u[:20], y[:20])

# Pelo gráfico da correlação cruzada, é possível ver que existem dois picos 
# principais, um em t=2 e outro em t=-3. Nesta análise, contrário à 
# autocorrelação, onde toma-se o ponto de mínimo apenas, nesta nova análise 
# tanto valores de máximo quanto de mínimo são igualmente importantes.
# Obs.: Veja isso no gráfico ruy, onde os pontos citados estão 
# marcados em vermelho.
t_max = np.argmax(ruy)
t_min = np.argmin(ruy)
plt.figure()
plt.plot(tuy, ruy, 'b-', label='$r_{uy}$', zorder=2)
plt.scatter(tuy[[t_min, t_max]], ruy[[t_min, t_max]],
            color='red', zorder=3)
plt.legend()
plt.grid(True, zorder=1)

# Atraso máximo do modelo estimado pela análise de correlação cruzada.
atraso = 3


# %%
# Preparação para aplicar o algoritmo de otimização, determinando as funções
# custo e os parãmetros.

# Determinando os parâmetros que serão utilizados para aplicar o algoritmo de
# otimização para determinar os parâmetros "ideais" para a rede neural,
# ou seja, o número de neurônios na camada oculta e a função de ativação dos
# neurônios.

#Primeiros N bits definem o número de neurônios na camada oculta (mínimo de 1)
#2 bits definem a função de ativação, ou seja, o número mínimo de bits é 3.
bits = 8

# Número de iterações executadas pelo algoritmo de otimização.
iteration_max = 50

#-----------------------------------------------------------------------------
# Funções custo:

bits = max(bits, 3) #garante o número mínimo de bits

# Esse valor é utilizado para fazer o custo da função custo depender não
# somente da métrica (RRSE, explicado adiante) mas também o número de
# neurônios que a compõe.
complexity_max = np.power(np.power(2,(bits-2)),2) #penalização quadratica

# Separa o conjunto de treinamento e teste
# (2/3 para treinamento e 1/3 para teste).
limite = int(0.66*len(u))
u_id = u[:limite]
y_id = y[:limite]

u_test = u[limite:]
y_test = y[limite:]

# Utilizou-se uma rede perceptron multi-layer, ou seja, uma rede neural que,
# inicialmente não aceita receber elementos com atraso (y(k-1), por exemplo,
# sendo utilizado como entrada do momento atual o valor de saída no instante
# anterior). Por este motivo, adaptou-se a rede, utilizando alguns neurônios
# de saída como neurônio de entrada. A função abaixo executa o treinamento
# da rede nessas condições.
def train_network(u, y, lag, hidden_layer_sizes, activation):
    u = np.array(u).tolist()
    y = np.array(y).tolist()
    
    input_id = []
    for k in range(1, lag+1):
        k_i = lag-k
        k_o = -k
        input_id.append(u[k_i:k_o])
        input_id.append(y[k_i:k_o])
    input_id = np.array(input_id).T

    neural = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                          alpha=1e-1,
                          max_iter=5000,
                          activation=activation,
                          solver='lbfgs')
    neural.fit(input_id,y[lag:])

    return neural


# Pelo mesmo motivo acima, a aplicação da rede no conjunto de dados também 
# precisa ser adaptado, sendo feito na função abaixo de forma análoga à
# função de treinamento.
def predict_network(u, y, lag, network_object):
    u = np.array(u).tolist()
    y = np.array(y).tolist()
    
    output = y[:lag]
    for k in range(lag, len(u)):
        input_sim = []
        for i in range(1, lag+1):
            input_sim.append(u[k-i])
            input_sim.append(output[k-i])
        input_sim = np.array(input_sim).reshape(1,-1)
        output_new_value = network_object.predict(input_sim)
        output.append(output_new_value[0])

    return np.array(output[lag:])


# Essa métrica foi utilizada para medir a qualidade da predição. Ela é
# "análoga" à métrica F1 porém aplicada à séries temporais de
# sistemas dinâmicos. Valores próximos de zero são ideais e valores iguais a
# zero indicam que o modelo sofreu "overfitting".
def rrse(y_base, y_sim):
    return np.sqrt(np.sum(np.power((y_base - y_sim),2)) / \
                   np.sum(np.power((y_base - np.mean(y_base)),2)))


# Como o número de neurônios é dada por um valor "inteiro" assim como a
# função de ativação (função 1, 2, 3, 4, etc) também, então utilizou-se uma
# codificação binária que era convertida para valores inteiros no momento
# de aplicar.
def bin2int(bit):
    return int(np.sum([(bit[::-1][i])*mt.pow(2,i) for i in range(len(bit))]))


# Função que traduz o código binário no número de neurônios da camada oculta.
def translate_hl_size(code):
    hl_len = bin2int(code) + 1
    return (hl_len,)


# Função que traduz o código binário na função de ativação utilizada.
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


# Função custo a ser minimizada. Ela pega o código binário, traduz eles para
# a função de ativação e número de neurônios na camada oculta, usa o conjunto
# de treinamento para trerinar essa rede. Aplicada a rede sobre o conjunto
# de teste, aplica o RRSE sobre essa predição para ser utilizada como
# função custo e este valor é multiplicado pela complexidade
# (valor proporcional ao número de neurônios do modelo), valorizando assim
# redes com baixo número de neurônios.

# Obs.: Valores que retornam erro fazem a rede ser descartada pelo algoritmo
# de otimização.
def cost_function(code):
    global u_id, y_id, u_test, y_test, atraso, complexity_max
    try:
        hl_size = translate_hl_size(code[:-2])
        act_func = translate_activation_function(code[-2:])
        
        complexity = np.power(np.prod(hl_size),2) / complexity_max
        
        cost = 0.5*complexity
        
        fit_metric_list = []
        for _ in range(3):
            ann_obj = train_network(u_id, y_id, atraso, hl_size, act_func)
            y_sim_test = predict_network(u_test, y_test, atraso, ann_obj)
            
            fit_metric = rrse(y_test[atraso:], y_sim_test)
            
            if (fit_metric <= 1e-15):
                raise Exception()
            
            fit_metric = fit_metric if (fit_metric >= 1) \
                                    else 2*np.power(fit_metric,4)
                                    
            fit_metric_list.append(fit_metric)
        
        cost += 0.5*np.max(fit_metric_list)
        
        return cost
        
    except:
        return np.nan


# %%
# Determinação da estrutura

# Nesta etapa, aplica-se o algoritmo de meta-heurística para otimizar a rede
# e determinar o número de neurônios e função de ativação.

# Ao final dessa etapa, um arquivo é salvo com o resultado da otimização.
# Isso é feito porque esse processo demora MUITO tempo para ser executado e
# tive problemas de acabar a eletricidade durante a madrugada, logo após o
# processo ter terminado, me obrigando a repetir tudo.

# ATENÇÃO:
# Descomentar e executar esse bloco do código leva muito tempo!
# (nos parâmetros iniciais levava cerca de 4 horas, nos parãmetros atuais,
# leva cerca de 1:30 horas)
# Aqui é recomendado executar o código no terminal e não na IDE
# (Spyder, VSCode, etc)

# import joblib as job
# from beecolpy import bin_abc
# from tqdm import tqdm
# import time

# print('Carregando população inicial do ABC.\n')
# abc_obj = bin_abc(cost_function,
#                   bits_count=bits,
#                   scouts=0.1,
#                   iterations=1)

# for i in tqdm(range(iteration_max)):
#     abc_obj.fit()
#     time.sleep(1)
#     with open('abc_neural.joblib', 'wb') as file:
#         job.dump(abc_obj, file)
#     time.sleep(1)
    
# print('\nFinalizado.')


# %%
# Validação do modelo

# Executa o treinamento da rede com os parâmetros otimizados e, em seguida,
# aplica a rede sobre os dados de validação, calculando-se o RRSE para
# verificar a qualidade do modelo.

# ATENÇÃO:
# Estre trecho do código depende do arquivo gerado durante a otimização.

# import joblib as job
# from beecolpy import bin_abc
# with open('abc_neural.joblib', 'rb') as file:
#     abc_obj = job.load(file)

# solution = abc_obj.get_solution()

# hl_sol = translate_hl_size(solution[:14])
# func_sol = translate_activation_function(solution[14:])
# ann_obj = train_network(u, y, atraso, hl_sol, func_sol)
# y_sim = np.empty(atraso)
# y_sim.fill(np.nan)
# y_sim = np.append(y_sim, predict_network(u_val, y_val, atraso, ann_obj))

# print(str(func_sol))
# print(str(hl_sol))
# print('{:.8f}'.format(rrse(y_val[atraso:], y_sim[atraso:])))

# plt.figure()
# plt.plot(y_val, 'b-', label='Dados', zorder=2)
# plt.plot(y_sim, 'r:', label='Modelo', zorder=3)
# plt.legend()
# plt.grid(True, zorder=1)


# %%
# Teste manual (usando os resultados já obtidos).

# Executa a mesma operação do bloco de código acima, contudo sem utilizar o
# arquivo gerado durante a otimização. Aqui pode-se colocar a função de
# ativação e o número de neurônios da camada oculta manualmente.
# Modelo obtido na otimização: (6,) 'tanh'

ann_obj = train_network(u, y, atraso, (6,), 'tanh')
y_sim = np.empty(atraso)
y_sim.fill(np.nan)
y_sim = np.append(y_sim, predict_network(u_val, y_val, atraso, ann_obj))

print('RRSE do modelo: {:.8f}'.format(rrse(y_val[atraso:], y_sim[atraso:])))

plt.figure()
plt.plot(y_val, 'b-', label='Dados', zorder=2)
plt.plot(y_sim, 'r:', label='Modelo', zorder=3)
plt.legend()
plt.grid(True, zorder=1)


# %%
# Análise de resíduos
residue = y_sim[atraso:] - y_val[atraso:]

rue, tue, rue_sup, rue_inf = correlation(u_val[atraso:21+atraso],
                                         residue[:21])

ree, tee, ree_sup, ree_inf = correlation(residue[:21])

plt.figure()
plt.plot(tue, rue, 'b-', label='$r_{ue}$', zorder=2)
plt.hlines([rue_inf, rue_sup], tue[0], tue[-1],
           color='black', linestyles=':', zorder=3)
plt.legend()
plt.grid(True, zorder=1)

plt.figure()
plt.plot(tee, ree, 'b-', label='$r_{ee}$', zorder=2)
plt.hlines([ree_inf, ree_sup], tee[0], tee[-1],
           color='black', linestyles=':', zorder=3)
plt.legend()
plt.grid(True, zorder=1)

