import sys
from sklearn.svm import SVC
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
    svm = SVC(kernel="linear")
    svm.fit(trainImages, trainLabels)
    tImages = valImages + testImages
    tLabels = valLabels + testLabels
    acc = help.testAccuracy(svm, tImages, tLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()
