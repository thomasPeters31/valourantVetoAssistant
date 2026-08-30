# --- Where this fits ---
# Final stage of the pipeline:
#   htmlScraper_VLR.py -> scrapes VLR.gg, writes mapResults.csv
#   analysis.py         -> reads mapResults.csv; this file imports its
#                          buildFrequencyTable() (win/played counts per
#                          team+map) and predictWinRate() (the fallback
#                          win-rate estimator built on top of those counts)
#   features.py         -> (this file) turns each mapResults.csv row into a
#                          numeric feature vector + win/loss label, ready
#                          for a model (X, Y arrays below)
#
# Run this file directly to build the full dataset and print its shape.

import csv
from analysis import buildFrequencyTable, predictWinRate, safeRate
import numpy as np

def canonicalOrder(teamA, teamB):
    # Puts the two team IDs in a fixed, match-independent order (lower ID
    # first) so the same matchup always produces features/labels in the
    # same team1/team2 order, regardless of which side VLR listed as "a".
    if teamA < teamB:
        return teamA, teamB
    else:
        return teamB, teamA
    
def pickedCanonicalTeam1(pickedBy, rawA, rawB):
    if pickedBy == "":
        return 0
    
    swapped = rawA >= rawB
    
    if not swapped:
        return 1 if pickedBy == "a" else -1
    else:
        return 1 if pickedBy == "b" else -1

def buildMapVocab(rows):
    maps = sorted(set(row["map"] for row in rows))
    return {mapName: i for i, mapName in enumerate(maps)}

def oneHotMap(mapName, mapVocab):
    vector = [0] * len(mapVocab)
    if mapName in mapVocab:
        vector[mapVocab[mapName]] = 1
    return vector

def buildFeatures(row, played, wins, teamPlayed, teamWins,
                   attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab):
    mapName = row["map"]
    rawA, rawB = row["teamID_a"], row["teamID_b"]
    team1, team2 = canonicalOrder(rawA, rawB)

    rate1 = predictWinRate(team1, mapName, played, wins, teamPlayed, teamWins)
    rate2 = predictWinRate(team2, mapName, played, wins, teamPlayed, teamWins)

    overall1 = safeRate(teamWins[team1], teamPlayed[team1])
    overall2 = safeRate(teamWins[team2], teamPlayed[team2])

    attackRate1 = safeRate(attackWon[team1], attackPlayed[team1])
    attackRate2 = safeRate(attackWon[team2], attackPlayed[team2])
    defenceRate1 = safeRate(defenceWon[team1], defencePlayed[team1])
    defenceRate2 = safeRate(defenceWon[team2], defencePlayed[team2])

    pickFeature = pickedCanonicalTeam1(row["pickedBy"], rawA, rawB)
    mapFeatures = oneHotMap(mapName, mapVocab)

    winnerID = row["teamID_a"] if row["winner"] == row["team_a"] else row["teamID_b"]
    label = 1 if winnerID == team1 else 0

    features = [rate1, rate2, overall1, overall2,
                attackRate1, attackRate2, defenceRate1, defenceRate2,
                pickFeature] + mapFeatures
    return features, label

def buildDataset(rows, played, wins, teamPlayed, teamWins, attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab):
    # Runs buildFeatures() over every row to assemble the full training
    # matrix (x) and label vector (y) for a model.
    x = []
    y = []

    for row in rows:
        features, label = buildFeatures(row, played, wins, teamPlayed, teamWins, attackWon, attackPlayed, defenceWon, defencePlayed, mapVocab)
        
        x.append(features)
        y.append(label)

    return np.array(x), np.array(y)

if __name__ == "__main__":
    with open('mapResults.csv', 'r') as f:
        rows = list(csv.DictReader(f))

    # Frequency/win-rate lookup tables built from the whole dataset (unlike
    # analysis.baseLineEvaluation, which trains only on a chronological
    # slice — this run isn't doing a train/test split).
    played, wins, teamPlayed, teamWins = buildFrequencyTable(rows)

    X, Y = buildDataset(rows, played, wins, teamPlayed, teamWins)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
