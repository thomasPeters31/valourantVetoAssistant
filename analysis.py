import csv
from collections import defaultdict
from collections import Counter

def pickWinRate():
     # Load every map row scraped by htmlScraper_VLR.py (one row per map
     # played, with who picked it and who won it).
     with open('mapResults.csv', 'r') as file:
         rows = list(csv.DictReader(file))

     print(f"Total number of maps: {len(rows)}")

     pickedWins = 0     # maps where the team that picked the map also won it
     pickedTotal = 0    # maps that were picked by a team (i.e. not a decider)
     deciderTotal = 0   # maps that were nobody's pick (the leftover/decider map)
     deciderWinsA = 0   # decider maps won by team A
     teamAWins = 0      # maps won by team A, picked or not

     for row in rows:
          pickedBy = row["pickedBy"]   # "a", "b", or "" for a decider map
          winner = row["winner"]
          teamA = row["team_a"]
          teamB = row["team_b"]

          if winner == teamA:
               teamAWins += 1

          # Decider maps have no pickedBy value — tally them separately
          # and skip the "did the picker win" logic below, since there's
          # no picker to credit.
          if pickedBy == "":
               deciderTotal += 1
               if winner == teamA:
                    deciderWinsA += 1
               continue


          pickedTotal += 1



          # Whichever side picked this map, did that same side go on to win it?
          if pickedBy == "a" and winner == teamA:
               pickedWins += 1
          elif pickedBy == "b" and winner == teamB:
               pickedWins += 1

     # % of picked maps where the picking team won their own pick.
     print(f"Picked maps: {pickedWins / pickedTotal * 100:.1f}%")
     # % of decider maps (nobody's pick) that team A won.
     print(f"Decider wins for A: {deciderWinsA / deciderTotal * 100:.1f}%")
     # Overall share of all maps (picked + decider) that team A won.
     print(f"Team A wins overall: {teamAWins}/{len(rows)} = {teamAWins / (len(rows)) * 100:.1f}%")

def buildFrequencyTable(rows):
     played = defaultdict(int)
     wins = defaultdict(int)
     teamPlayed = defaultdict(int)
     teamWins = defaultdict(int)

     for row in rows:
          mapName = row["map"]
          teamA = row["teamID_a"]
          teamB = row["teamID_b"]
          winner = row["winner"]

          played[(teamA, mapName)] += 1
          played[(teamB, mapName)] += 1
          teamPlayed[teamA] += 1
          teamPlayed[teamB] += 1

          if winner == row["team_a"]:
               wins[(teamA, mapName)] += 1
               teamWins[teamA] += 1
          else:
               wins[(teamB, mapName)] += 1
               teamWins[teamB] += 1

     return played, wins, teamPlayed, teamWins

def predictWinRate(teamID, mapName, played, wins, teamPlayed, teamWins, threshold=10, minTeamGames=5):
    key = (teamID, mapName)

    if played[key] >= threshold:
        return wins[key] / played[key]

    if teamPlayed[teamID] >= minTeamGames:
        return teamWins[teamID] / teamPlayed[teamID]

    return 0.5

def baseLineEvaluation():
     with open('mapResults.csv', 'r') as f:
          rows = list(csv.DictReader(f))
     
     rows.sort(key=lambda r: r["date"])  # Sort by date to simulate real-time prediction
     
     spiltPoint = int(len(rows) * 0.8)
     trainRows = rows[:spiltPoint]
     testRows = rows[spiltPoint:]
     
     print(f"Training on {len(trainRows)} rows, testing on {len(testRows)} rows")
     
     played, wins, teamPlayed, teamWins = buildFrequencyTable(trainRows)
     
     correctPredictions = 0
     totalPredictions = 0
     ties = 0
     
     for row in testRows:
          teamA = row["teamID_a"]
          teamB = row["teamID_b"]
          teamAName = row["team_a"]
          teamBName = row["team_b"]
          mapName = row["map"]
          winner = row["winner"]
          
          winRateA = predictWinRate(teamA, mapName, played, wins, teamPlayed, teamWins)
          winRateB = predictWinRate(teamB, mapName, played, wins, teamPlayed, teamWins)
          
          if winRateA == winRateB:
               ties += 1

          if winRateA > winRateB:
               predictedWinner = teamAName
          elif winRateB > winRateA:
               predictedWinner = teamBName
          else:
               predictedWinner = teamAName
          
          if predictedWinner == winner:
               correctPredictions += 1    
          
          totalPredictions += 1
          
          
          
     print(f"Baseline accuracy: {correctPredictions}/{totalPredictions} = {correctPredictions / totalPredictions * 100:.1f}%")
     print(f"Number of ties: {ties}/{totalPredictions} = {ties / totalPredictions * 100:.1f}%")

if __name__ == "__main__":
    pickWinRate()
     
