import numpy as np
from prettytable import PrettyTable
from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math

boston = load_boston()
X_y = np.column_stack([boston['data'], boston['target']])
np.random.seed(1)
np.random.shuffle(X_y)
X, y = X_y[:, :-1], X_y[:, -1]
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]
features = np.array(boston['feature_names'])
del X, y, X_y

# Add target to the end of the matrix
allInOneMatrix = np.concatenate((X_train, y_train[:, None]), axis=1)
featureLabels = np.append(features, np.array('PRICE'))  # add missing feature to features list
features = list(features)
# create the correlation matrix using numpy.corrcoef
# rowvar= False parameter means each row is a data and columns are features
# and calculates correlation and returns 14x14 matrix
correlationMatrix = np.corrcoef(allInOneMatrix, rowvar=False)

cmap = plt.cm.jet  # define the colormap for heatmap

# red color indicates high positive correlation
# blue color indicates high negative correlation
# ticklabels are  shown at edges. feature names
# sns.heatmap(correlationMatrix,  cmap=cmap, xticklabels=featureLabels, yticklabels=featureLabels)
correlationMatrix[:, -1]

correlations = [abs(elem) for elem in correlationMatrix[-1:][0][:-1]]
featureList = features
correlations, featureList = (list(t) for t in zip(*sorted(zip(correlations, featureList), reverse=True)))
# Correlations and features are sorted with same procedure.
most_correlated_feature = featureList[0]
least_correlated_feature = featureList[-1]
print('Most Correlated Feature: ', most_correlated_feature, '\nLeast Correlated Feature: ', least_correlated_feature)


# Given a feature name, returns the column index in features.
def getColumnIndex(features, feature):
    return features.index(feature)


def getRSS(matrix, allFeatures,  features, feature):
    colIndex = getColumnIndex(allFeatures, feature)

    # candidates are different data values of a certain feature.
    # All of them are tried as for finding the best RSS value.
    # Set operation is to reduce complexity with duplicates. Duplicates can be very common.
    candidatePoints = np.array(list(set(matrix[:, colIndex])))

    # Some initializations, guaranteed to change.

    bestRSS = 9999999
    rss = np.zeros(candidatePoints.shape)
    threshold = -1
    for index, point in enumerate(candidatePoints):

        # Data is split into 2 parts according to the 'feature' value it has.
        leftSide = matrix[matrix[:, colIndex] < point, :]
        rightSide = matrix[matrix[:, colIndex] >= point, :]

        # mean y values in the left split
        leftSideY = leftSide[:, -1]
        if (leftSideY.any()):
            leftMean = np.mean(leftSideY)
        else:
            leftMean = 0
        # mean y values in the right split
        rightSideY = rightSide[:, -1]
        if (rightSideY.any()):
            rightMean = np.mean(rightSideY)
        else:
            rightMean = 0

        # RSS values of left and right splits and summation.
        leftRSS = np.sum((leftSideY- leftMean) ** 2)
        rightRSS = np.sum((rightSideY - rightMean) ** 2)
        sumRSS = leftRSS + rightRSS
        rss[index] = sumRSS

        # If this rss value is less than overall best rss, then update the best rss and thresholds.
        if (sumRSS < bestRSS):
            threshold = point
            bestRSS = sumRSS
    return bestRSS, threshold, rss, candidatePoints


MCF_best, MCF_threshold, MCF_rss, MCF_points = getRSS(np.copy(allInOneMatrix), features, features, most_correlated_feature)
LCF_best, LCF_threshold, LCF_rss, LCF_points = getRSS(np.copy(allInOneMatrix), features, features, least_correlated_feature)


# plt.scatter(MCF_points, MCF_rss, color='orange')
# plt.scatter(LCF_points, LCF_rss)
# plt.title(f'Threshold-RSS Plot for feature {most_correlated_feature} and {least_correlated_feature}')
# plt.legend([f"n={most_correlated_feature}", f"n={least_correlated_feature}"])
# plt.xlabel('Values')
# plt.ylabel('RSS')
# plt.show()

# Node class are the tree nodes and stores the threshold
# and feature that best split the tree.
class Node():
    def __init__(self, feature):
        self.name = feature
        self.threshold = -1
        self.left = None
        self.right = None


def getBestRss(matrix, allFeatures, features):
    bestRss = 9999999
    bestThreshold = 9999999
    bestFeature = None
    for feature in features:

        rss, threshold, rssValues, candidatePoints = getRSS(matrix, allFeatures, features, feature)
        if (rss < bestRss):
            bestRss = rss
            bestThreshold = threshold
            bestFeature = feature

    return bestRss, bestThreshold, bestFeature


def createTree(tree, depth, maxDepth, allFeatures, splits):
    if (depth > maxDepth or tree.any() == 0):
        return None

    featuresToBeChosen = list(set(allFeatures) - set(splits))

    rss, threshold, feature = getBestRss(tree, allFeatures, featuresToBeChosen)


    node = Node(feature)
    node.threshold = threshold
    node.estimate = np.mean(tree[:, -1])
    index = getColumnIndex(allFeatures, feature)

    leftFilter = tree[:, index] < threshold
    # Selects the rows that feature column value is less than or equal to threshold
    rightFilter = tree[:, index] >= threshold
    # Selects the rows that feature column value is greater than threshold

    splits.append(feature)
    splitsSoFarL = splits.copy()  # Python architectural problems led me into this solution.
    splitsSoFarR = splits.copy()  # Independent subtrees were able to change their lists
    leftPart = tree[leftFilter, :]
    rightPart = tree[rightFilter, :]
    if (np.any(leftPart)):
        node.left = createTree(leftPart, depth + 1, maxDepth, allFeatures, splitsSoFarL)
    if (np.any(rightPart)):
        node.right = createTree(rightPart, depth + 1, maxDepth, allFeatures, splitsSoFarR)

    return node


# Build a tree with a stopping depth as 5
node = createTree(np.copy(allInOneMatrix), 1, 2, features, [])


def predict(data, features, node, sample):
    temp = node
    while (temp):
        feature = temp.name
        columnIndex = getColumnIndex(features, feature)
        value = sample[columnIndex]
        if (value < temp.threshold):
            a = data[data[:, columnIndex] < temp.threshold, :]
            if (not a.any()):
                break
            data = a
            temp = temp.left
        else:
            a = data[data[:, columnIndex] >= temp.threshold, :]
            if (not a.any()):
                break
            data = a
            temp = temp.right

    prediction = np.mean(data[:, -1])
    return prediction


def parser(X_train, S):
    np.random.shuffle(X_train)
    X_part = np.split(X_train, S, axis=0)  # X is split into 5

    X_train_sets = []

    X_train_set_1 = np.concatenate((X_part[1], X_part[2], X_part[3], X_part[4]), axis=0)
    X_train_set_2 = np.concatenate((X_part[0], X_part[2], X_part[3], X_part[4]), axis=0)
    X_train_set_3 = np.concatenate((X_part[0], X_part[1], X_part[3], X_part[4]), axis=0)
    X_train_set_4 = np.concatenate((X_part[0], X_part[1], X_part[2], X_part[4]), axis=0)
    X_train_set_5 = np.concatenate((X_part[0], X_part[1], X_part[2], X_part[3]), axis=0)

    X_train_sets.append(X_train_set_1)
    X_train_sets.append(X_train_set_2)
    X_train_sets.append(X_train_set_3)
    X_train_sets.append(X_train_set_4)
    X_train_sets.append(X_train_set_5)

    return X_train_sets, X_part


crossValidationSets, crossValidationTests = parser(allInOneMatrix, 5)

maxDepths = [i for i in range(3, 11)]
from sklearn.metrics import r2_score

results = [[0 for j in range(len(maxDepths))] for i in range(len(crossValidationSets))]

for i in range(len(crossValidationSets)):
    for di, maxDepth in enumerate(maxDepths):
        node = createTree(crossValidationSets[i], 1, maxDepth, features, [])
        predicts = np.zeros(crossValidationTests[i].shape[0])
        for ind, testRow in enumerate(crossValidationTests[i]):
            p = predict(crossValidationSets[i], features, node, testRow)
            predicts[ind] = p
        score = r2_score(crossValidationTests[i][:, -1], predicts)
        results[i][di] = score

means = np.mean(np.array(results), axis=0)
stds = np.std(np.array(results), axis=0)


table = PrettyTable()
table.field_names = ['Max depths'] + [str(dpth) for dpth in maxDepths]
table.add_row(["Cross Validation R2 Mean"] + [str(mn) for mn in means])
table.add_row(["Cross Validation R2 Std"] + [str(st) for st in stds])
print(table)

bestDepth = maxDepths[list(means).index(max(means))]
print('\n Best Max Depth is %s' % str(bestDepth))
data = np.copy(allInOneMatrix)
node = createTree(data, 1, bestDepth, features, [])
predicts = np.zeros(len(X_test))
for ind, sample in enumerate(X_test):
    p = predict(data, features, node, sample)
    predicts[ind] = p
score = r2_score(y_test, predicts)

print('score: ', score)
