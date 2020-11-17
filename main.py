'''
Projeto 1 - Sistemas Inteligentes
Perceptron - Diagnóstico de câncer de mama
Sarah R. L. Carneiro
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seaborn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

class Perceptron:

    def __init__(self, learningRate, numberOfTimes, weights):

        self.learningRate = learningRate  # taxa de apredisagem
        self.numberOfTimes = numberOfTimes  # numero de epocas
        self.W = weights  # pesos sinapticos
        self.activationFunc = self.unitStep  # funcao de ativacao

    def testPcn(self, X):

        return self.predict(X)

    def trainPcn(self, X, T, tipeOfTrain):

        error = []

        # treinamento de batch
        if (tipeOfTrain == 0):

            for i in range(self.numberOfTimes):
                O = self.predict(X)

                # W = W + lambda * X * (T - O)
                self.W += self.learningRate * np.dot(X.T, (T - O))

                error.append(Perceptron.calcError(O, T))

            return error

        # treinamento sequencial
        else:

            for i in range(self.numberOfTimes):
                for k in range(len(X)):
                    O = self.predict(X[k, :])
                    self.W += self.learningRate * np.dot(X[k, :][np.newaxis].T, (T[k, :] - O)[np.newaxis])

                O = self.predict(X)
                error.append(Perceptron.calcError(O, T))

            return error

    # f(XW - b)
    def predict(self, X):

        h = np.dot(X, self.W)
        return self.activationFunc(h)

    # funcao de ativacao - degrau unitario
    def unitStep(self, x):

        return np.where(x > 0, 1, 0)

    # Dataset disponível em: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    @staticmethod
    def loadData():

        file = open('breast-cancer-wisconsin.data', 'r')
        data = file.readlines()
        file.close()

        return data

    # Função que realiza o pré-processamento dos dados de entrada
    @staticmethod
    def dataTreatment(data):

        # atribui aos atributos faltantes, denotados por ?, o valor zero
        data = list(map(lambda i: i.replace('?', '0').split(','), data))
        data = np.array(data, dtype=int)

        # padrões de entrada
        X = data[:, 1:10]

        # normaliza os padrões de entrada para o intervalo [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        # adiciona o bias aos padroes de entrada
        X = np.concatenate((np.full((len(X), 1), -1.0), X), axis=1)

        # alvos
        labels = {2: 0, 4: 1}
        T = np.array([labels[x] for x in data[:, 10]]).reshape(699,1)

        return X, T

    @staticmethod
    def calcError(O, T):

        count = 0
        for i in range(len(T)):
            if (O[i] != T[i]):
                count += 1

        return count

    @staticmethod
    def calcAccuracy(O, T):

        error = Perceptron.calcError(O, T)
        return (1 - error/ len(T)) * 100

    @staticmethod
    def plotError(error, numberOfTimes):

        x = np.arange(0, numberOfTimes, 1)
        plt.plot(x, error, color='red')
        plt.title('Erro por época')
        plt.savefig('erro', bbox_inches='tight', dpi=300)
        plt.show()

    @staticmethod
    def plotConfusionMatrix(data, labels, output_filename):
        seaborn.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))

        plt.title("Matriz de confusão")

        seaborn.set(font_scale=1.4)
        ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt="d")

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # ax.set(ylabel="True Label", xlabel="Predicted Label")

        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

def main():
    print('\n\n===== Perceptron - Diagnóstico de câncer de mama =====\n\n')

    learningRate = float(input('Insira a taxa de aprendizagem:'))
    numberOfTimes = int(input('Insira o número de épocas:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    tipeOfTrain = int(input('Insira [0] para treinamento de lote e [1] treinamento sequencial:'))

    data = Perceptron.loadData()
    X, T = Perceptron.dataTreatment(data)

    # cria os conjuntos de treinamento e de teste
    idx = int(percentTrain * 699)

    trainSet = X[0:idx, :]
    testSet = X[idx+1:699, :]
    trainLabels = T[0:idx, :]
    testeLabels = T[idx+1:699, :]

    # pesos sinapticos
    np.random.seed(0)
    weights = np.random.normal(0, 0.01, (10, 1))

    p = Perceptron(learningRate=learningRate, numberOfTimes=numberOfTimes, weights=weights)
    error = p.trainPcn(trainSet, trainLabels, tipeOfTrain)
    O = p.testPcn(testSet)

    print('acurácia', Perceptron.calcAccuracy(O, testeLabels))

    conf_mat = confusion_matrix(testeLabels, O)
    print('Matriz de Confusão', conf_mat)

    Perceptron.plotError(error, numberOfTimes)
    Perceptron.plotConfusionMatrix(conf_mat, ['Tumor Benigno', 'Tumor Maligno'], 'matrizConfusao')

if __name__ == '__main__':
    main()