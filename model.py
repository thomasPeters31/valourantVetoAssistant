import numpy as np
import csv
from analysis import buildFrequencyTable
from features import buildDataset

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    z = X @ weights + bias
    return sigmoid(z)

if __name__ == "__main__":
    with open("mapResults.csv") as f:
        rows = list(csv.DictReader(f))

    played, wins, teamPlayed, teamWins = buildFrequencyTable(rows)
    X, y = buildDataset(rows, played, wins, teamPlayed, teamWins)

    weights = np.zeros(4)
    bias = 0.0

    preds = predict(X[:5], weights, bias)
    print(preds)