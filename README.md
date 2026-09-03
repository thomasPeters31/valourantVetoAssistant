# Valorant Veto Assistant

A machine learning model that helps managers estimate map win probability for Valorant matches, built to support veto decisions in tournament play.

**Input:** two teams and a map.
**Output:** estimated probability that the first team wins that map.

Trained on professional match data from VLR.gg, then tested in a tier 3 environment to see whether patterns learned at professional level transfer down, and whether they help with team success.

## Why

Veto happens under time pressure with incomplete information. Managers currently rely on memory and rough impressions of what an opponent is comfortable on. This replaces the guess with a number.

## Scope

Two models were built and compared: a historical-frequency baseline, and a
logistic regression implemented from scratch in numpy (gradient descent
hand-written, not from a library) to understand the mechanics rather than
just call `.fit()`. A PyTorch model was added later specifically to test
whether a more expressive architecture could do better.

## Data

Two scraping approaches were used, in sequence, as the sparsity problem
below was diagnosed and addressed:

**v1–v2 (global sweep):** every match across every region and tier,
scraped from VLR's general results feed. ~7,800 matches, 18,414 map
results. This is where the sparsity problem was first identified.

**v3 (event-based, region + tier filtered):** VLR's per-event match
listings, targeted at specific VCT-tier events by region. Event IDs are
auto-discovered from VLR's events listing (`vlr.gg/events/?region=X&tier=60`)
rather than hardcoded, filtered to 2023+ (the franchise era) and to the
target region's own events — excluding international events (Masters,
Champions, Lock-In), qualifiers, and developmental circuits (e.g. China
Evolution Series), all of which either mix in opponents outside the
target region or belong to a different competitive tier entirely.

Two datasets, joined on match ID:

- `vetoData.csv` — one row per veto step (ban, pick, or decider)
- `mapResults.csv` — one row per map played, with scores, attack/defence
  splits, team IDs, match date, and which team picked it

**Known limitation:** pre-2023 matches use a different VLR page template
and fail to parse (`parseTeam`'s `div.score` selector doesn't match). The
v3 pipeline avoids this by filtering to 2023+ at the source; it only
affects the legacy v1–v2 global sweep.

**Known limitation, resolved for single-region data, not for multi-region:**
the global sweep's team-map pairs were mostly sparse (three games or
fewer for the majority), because it swept every region and tier
indiscriminately. Region + tier filtering fixes this within a single
region. It does **not** fully fix it across multiple regions — see the v3
results below for why.

## Initial findings

| Measure | Rate | n |
|---|---|---|
| Picked maps won by the picking team | 51.8% | 18,414 |
| Team A wins (all maps) | 54.2% | 18,414 |
| Team A wins (decider maps only) | 54.7% | 18,414 |

Picking a map is worth about 2 percentage points over a coin flip —
smaller than expected, suggesting professional teams already veto
competently. Team A (however VLR happens to order the two teams in a
match) wins noticeably more often than team B, a structural artefact of
listing order rather than a skill effect. Every model below **canonically
re-orders each match's two teams by ID** to remove this leak, rather than
trusting VLR's listing order.

These numbers set a realistic ceiling for the model. Map outcomes carry
limited signal, so any claim of high predictive accuracy would more likely
indicate overfitting or leakage than a genuine result.

## Results: a systematic search for what actually helps (v1–v2, global sweep)

Every model below was evaluated with a **time-based train/test split**
(trained on the earliest matches by date, tested on the most recent) to
simulate genuine future prediction. The frequency-table baseline and the
logistic regression were both checked for stability across three different
split proportions (75/25, 80/20, 90/10), not just one.

| Model | Features | Test accuracy |
|---|---|---|
| Frequency baseline | Team-map rate, team overall rate (3-level fallback) | 55.6% (80/20) |
| v1: logistic regression | Same as baseline, numpy from scratch | 55.1–56.6% across splits |
| v1.1: + pick, map identity | + which team picked, one-hot map | 55.5% — no improvement |
| v1.2: + attack/defence rounds | + per-team attack/defence round win rate | 55.3% — no improvement |
| v2: PyTorch MLP | Same features as v1.2, one hidden layer (16 units, ReLU) | 55.6% — no improvement |

**Every model, regardless of features or architecture, lands in the same
55–57% band.**

### What was ruled out, and how

- **Pick and map-identity features** (v1.1) were added and verified to
  reach the model correctly (feature vector grew from 4 to 17 columns,
  confirmed via shape check) but produced no measurable improvement.
- **Attack/defence round rates** (v1.2) initially appeared to jump
  accuracy to ~68%. This was investigated rather than accepted at face
  value: the round-counting logic was verified against an independent
  from-scratch reimplementation (exact match), leakage was ruled out by
  confirming a team's historical rate is built entirely from other
  matches, and the result was checked across three splits. It held up
  every time — right up until a train/test accuracy comparison, intended
  as a final overfitting check, revealed the true cause: a variable
  mix-up meant the model had briefly been evaluated against its own
  training data rather than held-out matches. The corrected evaluation
  showed 55.3% — in line with everything else. Kept in this writeup
  because catching it was as instructive as the result itself: a check
  that passes is only meaningful if you're certain what data it actually
  used.
- **Model capacity** (v2) was tested by moving from a linear model to a
  small neural network capable of learning feature interactions a linear
  model can't represent. It converged to 55.6% — the same band. This
  points specifically at *data density*, not *model expressiveness*, as
  the limiting factor: a more capable model found nothing extra to
  exploit in the features available.

## Results: fixing the sparsity problem (v3)

The global sweep's density problem, measured directly: bucket every
team-map pair by how many games it has, as a share of all unique pairs.

| Dataset | ≤3 games | 11+ games | n (pairs) | Baseline accuracy (80/20) | Test n |
|---|---|---|---|---|---|
| v1–v2: global sweep, all regions/tiers | 57.5% | 13.1% | 7,302 | 55.6% | 3,683 |
| v3: Americas only, VCT tier, 2023+ | 25.7% | 49.2% | 179 | 58.5% | 195 |
| v3: 4 regions, VCT tier, 2023+ | 31.2% | 34.2% | 737 | 52.4% | 660 |

**Americas-only is the clean win.** Filtering to one region's partnered
league cuts the sparse-pair rate by more than half and pushes almost half
of all pairs into the well-populated 11+ bucket. The tie rate in
`baseLineEvaluation` (how often the frequency table has too little data
to commit to a prediction) dropped from 5.1% to 1.5% — a more reliable
signal than the accuracy figure itself, since it isn't sensitive to which
matches happened to land in a small test set. The 58.5% accuracy figure
should be read cautiously though: at n=195, the margin of error is
roughly ±7 points, so it's suggestive rather than conclusive on its own.

**Widening to 4 regions did not help, and the reason is structural, not a
bug.** The initial hypothesis — leftover qualifier/developmental events
leaking through the filter — was tested and partially confirmed (China
Evolution Series and LCQ events were found and excluded), but the deeper
cause survived that fix: **partnered team rosters change across seasons**
(e.g. Full Sense replacing Talon and Varrel being promoted via Ascension,
both into VCT Pacific for the 2026 season). Four regions across four
seasons of promotion/relegation produces 64 distinct team names, well
above what a single season's rosters would suggest, and a meaningful
share of them have almost no history before their promotion — reproducing
the sparsity problem the region filter was built to solve, through squad
turnover rather than dataset construction. This was verified by checking
individual team histories against public sources rather than assumed.

**Decision:** build forward on the Americas-only dataset (971 matches,
58.5% baseline), not the 4-region one. The 4-region attempt is a real,
documented result — evidence that per-team history depth matters more
than raw match count, and that roster churn is a genuine limitation of
this approach — kept here rather than discarded because it strengthens
the original sparsity finding rather than contradicting it.

## Roadmap

- **v3 — done.** Event-based, region + tier filtered scraping, auto-discovered
  via VLR's events listing. Fixed sparsity for single-region data;
  identified roster churn as a remaining limitation for multi-region data.
- **v3.1 — in progress.** Retest the v1.2 attack/defence round-rate
  features (previously no improvement, on sparse data) against the new
  dense Americas-only dataset. The feature and evaluation code already
  exists unchanged from v1.2; only the underlying data has changed.
- **v4** — sequence modelling over match history (recent form, streaks)
  rather than static career-average rates, once there's enough per-team
  data to make sequences meaningful.
- **v5** — web interface for the manager to query a matchup directly.