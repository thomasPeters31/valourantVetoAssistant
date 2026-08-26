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
     # Tally, per (team, map) pair, how many times that team played that
     # map and how many times they won it — a building block for later
     # per-team-per-map win-rate stats.
     played = defaultdict(int)
     wins = defaultdict(int)

     for row in rows:
          mapName = row["map"]
          winner = row["winner"]
          teamA = row["team_a"]
          teamB = row["team_b"]

          # Both teams in the match played this map, regardless of winner.
          played[(teamA, mapName)] += 1
          played[(teamB, mapName)] += 1

          if winner == row["team_a"]:
               wins[(teamA, mapName)] += 1
          else:
               wins[(teamB, mapName)] += 1

     # How many (team, map) pairs occur exactly once, twice, etc. — a quick
     # sanity check on sample size (e.g. "most teams have only played most
     # maps a handful of times") before trusting any per-team win rate.
     gameCounter = Counter(played.values())
     print(sorted(gameCounter.items()))

if __name__ == "__main__":
     with open('mapResults.csv', 'r') as file:
          rows = list(csv.DictReader(file))

     pickWinRate()
     buildFrequencyTable(rows)
