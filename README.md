# Valorant Veto Assistant

A machine learning model that helps managers estimate map win probability for Valorant matches, built to support veto decisions in tournament play.

**Input:** two teams and a map.
**Output:** estimated probability that the first team wins that map.

Trained on professional match data from VLR.gg, then tested in a tier 3 environment to see whether patterns learned at professional level transfer down, and whether they help with team success.

## Why

Veto happens under time pressure with incomplete information. Managers currently rely on memory and rough impressions of what an opponent is comfortable on. This replaces the guess with a number.

## v1 scope

Logistic regression implemented in numpy, with gradient descent written from scratch rather than using an ML library. The goal is to understand the mechanics, not to reach maximum accuracy. Benchmarked against a simple historical-frequency baseline.

## Roadmap

- **v2** — PyTorch, richer features
- **v3** — sequence modelling over match history
- **v4** — web interface