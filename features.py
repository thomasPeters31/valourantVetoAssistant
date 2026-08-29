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
from analysis import buildFrequencyTable, predictWinRate
import numpy as np

def canonicalOrder(teamA, teamB):
    # Puts the two team IDs in a fixed, match-independent order (lower ID
    # first) so the same matchup always produces features/labels in the
    # same team1/team2 order, regardless of which side VLR listed as "a".
    if teamA < teamB:
        return teamA, teamB
    else:
        return teamB, teamA

def buildFeatures(row, played, wins, teamPlayed, teamWins):
    # Turns one mapResults.csv row into a (features, label) pair for a
    # win-probability model: features describe both teams' map strength,
    # label says whether the canonical "team1" was the winner.
    mapName = row["map"]
    rawA, rawB = row["teamID_a"], row["teamID_b"]
    team1, team2 = canonicalOrder(rawA, rawB)

    # Each team's estimated win rate on this specific map (with the
    # per-map/per-team/50-50 fallback chain from analysis.predictWinRate).
    rate1 = predictWinRate(team1, mapName, played, wins, teamPlayed, teamWins)
    rate2 = predictWinRate(team2, mapName, played, wins, teamPlayed, teamWins)

    # Each team's win rate across all maps, as a map-agnostic skill signal.
    overallRate1 = teamWins[team1] / teamPlayed[team1] if teamPlayed[team1] > 0 else 0.5
    overallRate2 = teamWins[team2] / teamPlayed[team2] if teamPlayed[team2] > 0 else 0.5

    # row["winner"] holds the winning team's *name*, so compare it against
    # the name fields (team_a/team_b) to resolve the winner's *ID*.
    winnerID = row["teamID_a"] if row["winner"] == row["team_a"] else row["teamID_b"]
    label = 1 if winnerID == team1 else 0

    features = [rate1, rate2, overallRate1, overallRate2]
    return features, label

def buildDataset(rows, played, wins, teamPlayed, teamWins):
    # Runs buildFeatures() over every row to assemble the full training
    # matrix (x) and label vector (y) for a model.
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

    # Frequency/win-rate lookup tables built from the whole dataset (unlike
    # analysis.baseLineEvaluation, which trains only on a chronological
    # slice — this run isn't doing a train/test split).
    played, wins, teamPlayed, teamWins = buildFrequencyTable(rows)

    X, Y = buildDataset(rows, played, wins, teamPlayed, teamWins)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
