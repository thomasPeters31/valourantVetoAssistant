import csv
from analysis import buildFrequencyTable, predictWinRate
import numpy as np

def canonicalOrder(teamA, teamB):
    if teamA < teamB:
        return teamA, teamB
    else:
        return teamB, teamA

def buildFeatures(row, played, wins, teamPlayed, teamWins):
    mapName = row["map"]
    rawA, rawB = row["teamID_a"], row["teamID_b"]
    team1, team2 = canonicalOrder(rawA, rawB)
    
    rate1 = predictWinRate(team1, mapName, played, wins, teamPlayed, teamWins)
    rate2 = predictWinRate(team2, mapName, played, wins, teamPlayed, teamWins)
    
    overallRate1 = teamWins[team1] / teamPlayed[team1] if teamPlayed[team1] > 0 else 0.5
    overallRate2 = teamWins[team2] / teamPlayed[team2] if teamPlayed[team2] > 0 else 0.5
    
    winnerID = row["teamID_a"] if row["winner"] == rawA else row["teamID_b"]
    label = 1 if winnerID == team1 else 0
    
    features = [rate1, rate2, overallRate1, overallRate2]
    return features, label

def buildDataset(rows, played, wins, teamPlayed, teamWins):
    x = []
    y = []
    
    for row in rows:
        features, label = buildFeatures(row, played, wins, teamPlayed, teamWins)
        x.append(features)
        y.append(label)
        
    return np.array(x), np.array(y)
    

if __name__ == "__main__":
    with open('mapResults.csv', 'r') as f:
        rows = list(csv.DictReader(f))

    played, wins, teamPlayed, teamWins = buildFrequencyTable(rows)
    
    X, Y = buildDataset(rows, played, wins, teamPlayed, teamWins)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")