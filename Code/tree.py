import sys
from sklearn.tree import DecisionTreeClassifier as DTC
import helper as help

def tuneHyperParams(trainImages, trainLabels, valImages, valLabels):
    accuracyArr = []
    paramsArr = []
    for max_depth in range(10, 15):
        for min_samples_split in [2,4,6,8]:
            params = {
                "max_depth":max_depth,
                "min_samples_split":min_samples_split,
                "criterion":"entropy"
            }
            tree = DTC(**params)
            tree.fit(trainImages, trainLabels)
            accuracy = help.testAccuracy(tree, valImages, valLabels)
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
    print(accuracyArr)
    print(paramsArr)
    tree = DTC(**paramsArr[0])
    tree.fit(trainImages, trainLabels)
    tImages = valImages + testImages
    tLabels = valLabels + testLabels
    acc = help.testAccuracy(tree, tImages, tLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()
