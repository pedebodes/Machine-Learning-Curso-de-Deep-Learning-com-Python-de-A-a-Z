import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


previsores = pd.read_csv('../datasets/entradas-breast.csv')
classe = pd.read_csv('../datasets/saidas-breast.csv')


def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    # construindo a primeira camada oculta e a definição da camada de entrada
    # aqui eu posso colocar um for para criar quantas camadas ocultas eu queira
    classificador = Sequential()

    #
    for camada in range(3):
        classificador.add(Dense(units = neurons,
                                activation = activation,
                                kernel_initializer = kernel_initializer,
                                input_dim = 30))
    # camada de saída
    classificador.add(Dropout(0.2))
    # aqui usamos a função sigmoid por que o esultado esperado é 0 ou 1
    classificador.add(Dense(units= 1, activation='sigmoid'))
    # configurando a rede neural
    classificador.compile(optimizer = optimizer, loss = loos,metrics = ['binary_accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criarRede)

parametros = {'batch_size':[10,30],
             'epochs': [1],
             'optimizer': ['adam', 'sgd'],
             'loos' : ['binary_crossentropy', 'hinge'],
             'kernel_initializer' : ['random_uniform', 'normal'],
             'activation' : ['relu', 'tanh'],
             'neurons':[16,8]}


grid_search = GridSearchCV(estimator=classificador,
                          param_grid = parametros,
                          scoring = 'accuracy',
                          cv = 5)


grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros)
print(melhor_precisao)
