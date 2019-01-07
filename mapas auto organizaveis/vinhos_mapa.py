#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:48:43 2019

@author: alison
"""

from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar

base = pd.read_csv('../datasets/wines.csv')

X = base.iloc[:, 1:14].values
y = base.iloc[:,0].values

# normalizando os valores entre 0 e 1
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# construção do mapa auto-organizavel 
# os valores de x e y foram alcançados pelo cálculo
# 5 raiz de N ou seja 5 raiz de 178, o resultado deve ser colocado em uma raiz quadrada
# por que queremos transformar em uma matriz, ou seja o resultado aqui é 65,65
# raiz de 65 é aproximada de 8 

# x = linhas e y = colunas, input_len = numero de entradas, sigma = ao raio
# partindo do centroide, learning_rate taxa de atualização dos pesos
# random_seed é a semente para a geração dos primeiros valores dos pesos
som = MiniSom(x = 8, y = 8, input_len=13, sigma= 1.0, learning_rate=0.5, random_seed=2) 
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)
# aqui finaliza a criação do mapa

# analisando o mapa
# visualizando os pesos criados
som._weights

# valores do mapa auto-organizavél
som._activation_map

# visualizando quantas vezes cada neurônio foi selecionado como o BMU (o melhor centroide)
q = som.activation_response(X)

# Visualizando o quanto um neurônio é parecido com seus vizinhos
# quanto mais escuro o neurônio for significa que mais parecido ele é dos seus 
# vizinhos já o oposto acontece quando por ser mais claro e esses diferentes podem 
# não ser muito confiavéis por serem muito diferentes dos seus vizinhos
pcolor(som.distance_map().T)
colorbar()
