import csv

def pickWinRate():
     with open('mapResults.csv', 'r') as file:
         rows = list(csv.DictReader(file))
         
     print(f"Total number of maps: {len(rows)}")
         
     pickedWins = 0
     pickedTotal = 0
     deciderTotal = 0
     deciderWinsA = 0
     teamAWins = 0
         
     for row in rows[0:]:
          pickedBy = row["pickedBy"]
          winner = row["winner"]
          teamA = row["team_a"]
          teamB = row["team_b"]
                    
          if winner == teamA:
               teamAWins += 1
                    
          if pickedBy == "":
               deciderTotal += 1
               if winner == teamA:
                    deciderWinsA += 1
               continue
          
          
          pickedTotal += 1
          
         
          
          if pickedBy == "a" and winner == teamA:
               pickedWins += 1
          elif pickedBy == "b" and winner == teamB:
               pickedWins += 1

     print(f"Picked maps: {pickedWins / pickedTotal * 100:.1f}%")
     print(f"Decider wins for A: {deciderWinsA / deciderTotal * 100:.1f}%")
     print(f"Team A wins overall: {teamAWins}/{len(rows)} = {teamAWins / (len(rows)) * 100:.1f}%")
     
if __name__ == "__main__":
     pickWinRate()