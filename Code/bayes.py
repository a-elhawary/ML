from sklearn.naive_bayes import GaussianNB as GNB
import sys
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
    print("training model...")
    gnb = GNB()
    gnb.fit(trainImages, trainLabels)
    print("Model Trained...")
    print("Accuracy over both Testing and Validation sets is:")
    tImages = valImages + testImages
    tLabels = valLabels + testLabels
    acc = help.testAccuracy(gnb, tImages, tLabels, True, size)
    print(acc)


if __name__ == "__main__":
    main()
