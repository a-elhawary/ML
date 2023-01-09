import sys
from sklearn.svm import SVC
import helper as help

def tuneHyperParams(trainImages, trainLabels, valImages, valLabels):
    paramsArr = []
    accuracyArr = []
    # Try linear SVM
    params = {"kernel":"linear"}
    svm = SVC(**params)
    svm.fit(trainImages, trainLabels)
    accuracyArr.append(help.testAccuracy(svm, valImages, valLabels))
    paramsArr.append(params)

    # Try some polynomial kernel SVMS
    for degree in range(2, 6):
        params = {"kernel":"poly", "degree":degree}
        svm = SVC(**params)
        svm.fit(trainImages, trainLabels)
        accuracy = help.testAccuracy(svm, valImages, valLabels)
        help.insertSorted(accuracy, params, accuracyArr, paramsArr)

    # Try some polynomial kernel SVMS
    gammas = ["scale", "auto"]
    for gamma in gammas:
        params = {"kernel":"rbf", "gamma":gamma}
        svm = SVC(**params)
        svm.fit(trainImages, trainLabels)
        accuracy = help.testAccuracy(svm, valImages, valLabels)
        help.insertSorted(accuracy, params, accuracyArr, paramsArr)
    return accuracyArr, paramsArr


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
    accuracyArr, paramsArr = tuneHyperParams(trainImages, trainLabels, valImages, valLabels)
    print(paramsArr)
    print(accuracyArr)
    svm = SVC(**paramsArr[0])
    svm.fit(trainImages, trainLabels)
    acc = help.testAccuracy(svm, testImages, testLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()
