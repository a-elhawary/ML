import sys
from sklearn.neural_network import MLPClassifier as MLP
import helper as help


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
    mlp = MLP()
    mlp.fit(trainImages, trainLabels)
    tImages = valImages + testImages
    tLabels = valLabels + testLabels
    acc = help.testAccuracy(mlp, tImages, tLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()