import sys
from sklearn.neural_network import MLPClassifier as MLP
import helper as help
import bisect
import numpy as np

def tuningHyperParams(trainImages, trainLabels, valImages, valLabels):
    hidden_layer_sizes_sum = [(100,), (200,), (50,50,)]
    activations = ['logistic', 'tanh', 'relu']
    solver = 'sgd'
    alphas = [0.0001, 0.001, 0.002, 0.1]
    learning_rates = [x / 1000.0 for x in range(10, 30 ,5)]
    random_state = 2024
    i = 0
    paramsArr = []
    accuracyArr = []
    for hidden_layer_sizes in hidden_layer_sizes_sum:
        for activation in activations:
            for alpha in alphas:
                for learning_rate_init in learning_rates:
                    print("Training Model: " + str(i + 1))
                    params = {
                            'hidden_layer_sizes': hidden_layer_sizes,
                            'alpha':alpha,
                            'learning_rate_init': learning_rate_init,
                            'solver': solver,
                            'random_state':random_state
                    }
                    mlp = MLP(**params)
                    mlp.fit(trainImages, trainLabels)
                    print("Test Accuracy")
                    accuracy = help.testAccuracy(mlp, valImages, valLabels)
                    help.insertSorted(accuracy, params, accuracyArr, paramsArr)
                    i += 1
    return accuracyArr[:5], paramsArr[:5]


def main():
    dataset = "digitdata"
    size = (28, 28)
    n = (5000, 1000, 1000)
    if len(sys.argv) >= 2 and sys.argv[1] == "face":
        dataset = "facedata"
        size = (60, 70)
        n = (451, 301, 150)
    print("Loading datasets...")
    trainImages, trainLabels = help.getImages(n[0], "training", dataset, size)
    valImages, valLabels = help.getImages(n[1], "validation", dataset, size)
    testImages, testLabels = help.getImages(n[2], "test", dataset, size)
    print("Datasets Loaded...")
    accuracyArr, paramsArr = tuningHyperParams(trainImages, trainLabels, valImages, valLabels)
    print(accuracyArr)
    print(paramsArr)
    mlp = MLP(random_state=2024)
    mlp.fit(trainImages, trainLabels)
    tImages = valImages + testImages
    tLabels = valLabels + testLabels
    acc = help.testAccuracy(mlp, tImages, tLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()
