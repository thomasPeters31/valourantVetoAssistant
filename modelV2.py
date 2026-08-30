import torch
import torch.nn as nn
import csv
from analysis import buildFrequencyTable
from features import buildDataset, buildMapVocab

class VetoModel(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        self.layer1 = nn.Linear(numFeatures, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    print(torch.__version__)

    with open("mapResults.csv") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: r["date"])
    splitPoint = int(len(rows) * 0.8)
    trainRows = rows[:splitPoint]
    testRows = rows[splitPoint:]

    (played, wins, teamPlayed, teamWins,
     attackWon, attackPlayed, defenceWon, defencePlayed) = buildFrequencyTable(trainRows)
    mapVocab = buildMapVocab(trainRows)

    XTrain, yTrain = buildDataset(trainRows, played, wins, teamPlayed, teamWins,
                                   attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab)
    XTest, yTest = buildDataset(testRows, played, wins, teamPlayed, teamWins,
                                 attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab)

    XTrainTensor = torch.tensor(XTrain, dtype=torch.float32)
    yTrainTensor = torch.tensor(yTrain, dtype=torch.float32)
    XTestTensor = torch.tensor(XTest, dtype=torch.float32)
    yTestTensor = torch.tensor(yTest, dtype=torch.float32)

    model = VetoModel(numFeatures=XTrain.shape[1])
    print(model)

    testOutput = model(XTrainTensor[:5])
    print(testOutput)