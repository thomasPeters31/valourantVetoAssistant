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

def train(X, Y, learningRate=0.1, epochs=5000):
    weights = np.zeros(X.shape[1])
    bias = 0.0
    
    for epoch in range(epochs):
        preds = predict(X, weights, bias)
        loss = computeLoss(Y, preds)
        gradWeights, gradBias = computeGradients(X, Y, preds)
        
        weights -= learningRate * gradWeights
        bias -= learningRate * gradBias
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss: .4f}")
    
    return weights, bias

if __name__ == "__main__":
    with open("mapResults.csv") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: r["date"])
    splitPoint = int(len(rows) * 0.9)
    trainRows = rows[:splitPoint]
    testRows = rows[splitPoint:]

    played, wins, teamPlayed, teamWins = buildFrequencyTable(trainRows)

    XTrain, yTrain = buildDataset(trainRows, played, wins, teamPlayed, teamWins)
    XTest, yTest = buildDataset(testRows, played, wins, teamPlayed, teamWins)

    weights, bias = train(XTrain, yTrain, epochs=5000)

    testPreds = predict(XTest, weights, bias)
    predictedLabels = (testPreds >= 0.5).astype(int)

    accuracy = np.mean(predictedLabels == yTest)
    print(f"Test accuracy: {accuracy * 100:.1f}%")