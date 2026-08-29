import numpy as np
import csv
from analysis import buildFrequencyTable
from features import buildDataset

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    z = X @ weights + bias
    return sigmoid(z)

def computeLoss(y, preds):
    epsilon = 1e-15
    preds = np.clip(preds, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

def computeGradients(X, Y, preds):
    n = len(Y)
    error = preds - Y
    gradWeights = X.T @ error / n
    gradBias = np.mean(error)
    return gradWeights, gradBias

if __name__ == "__main__":
    with open("mapResults.csv") as f:
        rows = list(csv.DictReader(f))

    played, wins, teamPlayed, teamWins = buildFrequencyTable(rows)
    X, y = buildDataset(rows, played, wins, teamPlayed, teamWins)

    weights = np.zeros(4)
    bias = 0.0

    preds = predict(X, weights, bias)
    loss = computeLoss(y, preds)
    
    gradWeights, gradBias = computeGradients(X, y, preds)
    print(gradWeights)
    print(gradBias)