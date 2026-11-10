# Data

The raw file is not stored in this repository.

1. Download the dataset from Kaggle: **[FIFA World Cup 2026 player performance](KAGGLE_LINK)**
   (licence: KAGGLE_LICENCE).
2. Put the CSV at `data/raw/fifa_world_cup_2026_player_performance.csv`.
3. `make pipeline` checks the file against the SHA-256 below and warns if it differs from the
   file the committed results were produced with:

```
4ed9cc4b0e05b38f839b8411c1926974fc7d2477ac30b1948f5c0ea5b0d32632
```

## Contents

One row per player per match: 54,600 rows and 75 columns, covering 1,248 players
(48 squads of 26) across 1,050 matches dated between 11 June and 31 July 2026. The columns hold player
attributes (age, position, market value, ...), match context (date, opponent, stage, score)
and per-match statistics (minutes, goals, xG, passes, tackles, distance, ...), plus a
`player_rating` between 0 and 10. That rating is the forecasting target.

## The data is synthetic

The audit in [`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb) shows that
the data does not describe a real tournament:

- teams play 33–66 matches each (a real World Cup has at most 8 per team);
- in 521 cases a team plays two matches on the same day;
- matches labelled "Final" appear in seven of the eight weeks, mixed in with group matches;
- player names are recombinations of real first and last names;
- the rating is almost fully determined by synthetic composite scores (R² 0.86), but only
  weakly by the recorded events such as goals, passes and tackles (R² 0.22).

The project therefore uses the data as a sandbox for methodology: leakage-safe features,
honest evaluation and calibrated uncertainty. The football conclusions do not transfer to
real matches.
