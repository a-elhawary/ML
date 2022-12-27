from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pyplot as plt
import sys
import helper

def main():
    dataset= "digitdata"
    size=(28,28)
    n = (5000,1000,1000)
    if len(sys.argv) >= 2 and sys.argv[1] == "face":
        dataset = "facedata"
        size = (60, 70)
        n = (451, 301, 150)
    print("Loading datasets...")
    trainImages, trainLabels=helper.getImages(n[0], "training", dataset, size)
    valImages, valLabels=helper.getImages(n[1], "validation",dataset, size)
    testImages, testLabels=helper.getImages(n[2], "test",dataset, size)
    print("Datasets Loaded...")
    print("Tuning Hyperparameters...")
    maxK = 0
    maxAccuracy = 0
    maxDistance = 0
    kValues = []
    euclidean = []
    manhattan = []
    for k in range(12):
        for i in range(2):
            knn = KNN(n_neighbors=k+2, p=i+1)
            knn.fit(trainImages, trainLabels)
            accuracy = helper.testAccuracy(knn, valImages, valLabels)
            if accuracy > maxAccuracy:
                maxAccuracy = accuracy
                maxK = k+2
                maxDistance = i+1
            distance = " Manhattan "
            if i == 1:
                distance = " Euclidean "
                euclidean.append(accuracy)
                kValues.append(k+2)
            else:
                manhattan.append(accuracy)
            print("K: " + str(k+2) + distance + str(accuracy) + "%")
    plt.plot(kValues, euclidean)
    plt.plot(kValues, manhattan)
    plt.show()
    print("Using K="+str(maxK))
    print("Using Distance="+str(maxDistance))
    knn = KNN(n_neighbors=maxK)
    knn.fit(trainImages, trainLabels)
    acc = helper.testAccuracy(knn, testImages, testLabels, True, size)
    print(acc)

if __name__ == "__main__":
    main()
