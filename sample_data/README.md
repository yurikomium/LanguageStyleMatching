# Sample Data

This directory contains sample dialogue data for demonstrating the Language Style Matching (LSM) algorithm.

## Data Format

Each CSV file contains the following columns:
- `speaker`: Speaker identifier ("female" or "male")
- `text`: Utterance text (in Japanese)
- `start`: Utterance start time (in seconds)

## File Naming Convention

File names follow this format:
```
{SessionID}_{Round}_{Group}_A_F{FemaleID}_M{MaleID}_concat.csv
```

Examples:
- `01_1_1_A_F001_M001_concat.csv`: Session 01, Round 1 (first), dialogue between Female 001 and Male 001
- `01_2_1_A_F001_M001_concat.csv`: Session 01, Round 2 (second), same pair's dialogue
- `02_1_1_A_F002_M002_concat.csv`: Session 02, Round 1, dialogue between Female 002 and Male 002

## Note

These are fictional demonstration data, not actual dialogue data.
They are intended for verifying the LSM algorithm implementation and functionality.
