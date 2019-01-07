import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('../datasets/entradas-breast.csv')
classe = pd.read_csv('../datasets/saidas-breast.csv')

def criarRede(**parametros_rede ):
    # construindo a primeira camada oculta e a definição da camada de entrada
    # aqui eu posso colocar um for para criar quantas camadas ocultas eu queira

    classificador = Sequential()

    #
    for camada in range(parametros_rede['camadas_ocultas']):
        classificador.add(Dense(units = parametros_rede['units_camadas_ocultas'],
                                activation = parametros_rede['activation_camadas_ocultas'],
                                kernel_initializer = parametros_rede['kernel_initializer'],
                                input_dim = parametros_rede['input_dim']))
    # camada de saída
    classificador.add(Dropout(parametros_rede['dropout']))
    # aqui usamos a função sigmoid por que o esultado esperado é 0 ou 1
    classificador.add(Dense(units = parametros_rede['units_camada_saida'],
                            activation = parametros_rede['activation_camada_saida']))
    otimizador = keras.optimizers.Adam(lr = parametros_rede['lr'], decay = parametros_rede['decay'],
                                       clipvalue = parametros_rede['clipvalue'])
    # configurando a rede neural
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',metrics = ['binary_accuracy'])

    return classificador

parametros_rede = {'camadas_ocultas': 8, 'units_camadas_ocultas' : 8, 'activation_camadas_ocultas' : 'relu',
              'kernel_initializer' : 'normal', 'input_dim' : 30, 'dropout' : 0.2, 'units_camada_saida' : 1,
                        'activation_camada_saida' : 'sigmoig','lr' : 0.0001, 'decay' : 0.0001, 'clipvalue' : 0.5}

classificador = KerasClassifier(build_fn=criarRede, epochs = 100, batch_size = 10)

resultados = cross_val_score(estimator=classificador, X = previsores, y = classe, cv = 10, scoring='accuracy')                        
