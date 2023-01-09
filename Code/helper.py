import samples
import numpy as np
import random
import math
from matplotlib import pyplot as plt


def insertSorted(accuracy, params, accuracyArr, paramsArr):
    idx = -1
    for k in range(len(accuracyArr)):
        if accuracy > accuracyArr[k]:
            idx = k
            break
    if idx == -1:
        accuracyArr.append(accuracy)
        paramsArr.append(params)
    else:
        accuracyArr.insert(idx, accuracy)
        paramsArr.insert(idx, params)


def getImages(IMG_SIZE, filename, dataset, size):
    print("../data/"+dataset+"/"+filename+"images")
    filePath = "../data/"+dataset+"/"+filename+"images"
    data = samples.loadDataFile(filePath, IMG_SIZE, size[0], size[1])
    images = []
    for i in range(len(data)):
        image = []
        for row in data[i].getPixels():
            for pixel in row:
                image.append(pixel)
        images.append(image)
    labelPath = "../data/"+dataset+"/"+filename+"labels"
    labels = samples.loadLabelsFile(labelPath, IMG_SIZE)
    return images, labels


def testAccuracy(knn, images, labels, returnViz=False, shape=(28, 28)):
    predictions = knn.predict(images)
    count = 0
    vizImages = [[], []]
    vizLabels = [[], []]
    for i in range(len(predictions)):
        idx = 1
        if predictions[i] == labels[i]:
            count += 1
            idx = 0
        if returnViz:
            image = np.array(images[i])
            image = image * 127
            image = image.reshape((shape[1], shape[0]))
            vizImages[idx].append(image)
            vizLabels[idx].append(predictions[i])
    if returnViz:
        # Get Random Samples from both correct and incorrect images
        numShown = 16  # numShown should have an integer square root
        showImages = []
        showLabels = []
        for i in range(int(numShown/2)):
            idx = random.randint(0, len(vizImages[0]) - 1)
            showImages.append(vizImages[0][idx])
            showLabels.append(vizLabels[0][idx])
        for i in range(int(numShown/2), numShown):
            idx = random.randint(0, len(vizImages[1]) - 1)
            showImages.append(vizImages[1][idx])
            showLabels.append(vizLabels[1][idx])
        axisSize = int(math.sqrt(numShown))
        _, axis = plt.subplots(axisSize, axisSize)
        k = 0
        for i in range(axisSize):
            for j in range(axisSize):
                axis[i, j].imshow(showImages[k])
                axis[i, j].set_title(showLabels[k])
                axis[i, j].set_yticklabels([])
                axis[i, j].set_xticklabels([])
                k += 1
        plt.show()
    return count * 1.0 / len(predictions) * 100
