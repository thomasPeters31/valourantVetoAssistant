# --- Where this fits ---
# Modeling stage of the pipeline, built on top of the first three files:
#   htmlScraper_VLR.py -> scrapes VLR.gg, writes mapResults.csv
#   analysis.py         -> reads mapResults.csv; this file imports its
#                          buildFrequencyTable() to build win-rate lookups
#                          from the training rows only (no test leakage)
#   features.py         -> this file imports its buildDataset() to turn
#                          rows + those lookups into numeric X/Y arrays
#   model.py             -> (this file) trains a logistic regression model
#                          from scratch on X/Y and reports test accuracy —
#                          the "real" model to compare against analysis.py's
#                          simple baseLineEvaluation() baseline
#
# Run this file directly to train on the earliest 90% of matches
# (chronologically) and evaluate on the most recent 10%.

import numpy as np
import csv
from analysis import buildFrequencyTable, safeRate
from features import buildDataset, buildMapVocab

def sigmoid(z):
    # Squashes any real-valued score into a (0, 1) probability.
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    # Linear combination of features -> sigmoid -> predicted win probability
    # for "team1" (see features.buildFeatures for what team1/label mean).
    z = X @ weights + bias
    return sigmoid(z)

def computeLoss(y, preds):
    # Binary cross-entropy: penalizes confident-but-wrong predictions much
    # more heavily than uncertain ones.
    epsilon = 1e-15
    # log(0) is -inf, so clip predictions away from the exact 0/1 edges
    # before taking the log.
    preds = np.clip(preds, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

def computeGradients(X, Y, preds):
    # Gradient of the cross-entropy loss w.r.t. weights and bias, for a
    # sigmoid model — used to step the weights/bias downhill each epoch.
    n = len(Y)
    error = preds - Y
    gradWeights = X.T @ error / n
    gradBias = np.mean(error)
    return gradWeights, gradBias

def train(X, Y, learningRate=0.1, epochs=5000):
    # Plain batch gradient descent: start from zero weights/bias and
    # repeatedly nudge them in the direction that reduces the loss.
    weights = np.zeros(X.shape[1])
    bias = 0.0

    for epoch in range(epochs):
        preds = predict(X, weights, bias)
        loss = computeLoss(Y, preds)
        gradWeights, gradBias = computeGradients(X, Y, preds)

        weights -= learningRate * gradWeights
        bias -= learningRate * gradBias

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}: loss = {loss: .4f}")

    return weights, bias

if __name__ == "__main__":
    with open("mapResults.csv") as f:
        rows = list(csv.DictReader(f))

    # Chronological 90/10 split, same idea as analysis.baseLineEvaluation:
    # train on older matches, test on newer ones, so the model is never
    # evaluated on data from before its "training cutoff".
    rows.sort(key=lambda r: r["date"])
    splitPoint = int(len(rows) * 0.8)
    trainRows = rows[:splitPoint]
    testRows = rows[splitPoint:]

    # Win-rate lookups built only from training rows, then reused to build
    # features for both splits — the test set's features still only reflect
    # what was knowable at training time, avoiding leakage.
    (played, wins, teamPlayed, teamWins,
    attackWon, attackPlayed, defenceWon, defencePlayed) = buildFrequencyTable(trainRows)
    mapVocab = buildMapVocab(trainRows)

    XTrain, yTrain = buildDataset(trainRows, played, wins, teamPlayed, teamWins, attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab)
    XTest, yTest = buildDataset(testRows, played, wins, teamPlayed, teamWins, attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab)

    weights, bias = train(XTrain, yTrain, epochs=15000)

    # Predicted probabilities >= 0.5 count as a predicted "team1 win" (see
    # features.buildFeatures for how team1/label are defined).
    testPreds = predict(XTest, weights, bias)
    predictedLabels = (testPreds >= 0.5).astype(int)

    print(XTrain.shape)
    print(len(mapVocab), mapVocab)
    
    accuracy = np.mean(predictedLabels == yTest)
    print(f"Test accuracy: {accuracy * 100:.1f}%")
    
    trainPreds = predict(XTrain, weights, bias)
    trainPredictedLabels = (trainPreds >= 0.5).astype(int)
    trainAccuracy = np.mean(trainPredictedLabels == yTrain)
    print(f"Train accuracy: {trainAccuracy * 100:.1f}%")    