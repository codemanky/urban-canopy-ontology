# Reliable Agents With Ontology

This project demonstrates the value of using an OWL/SKOS ontology to normalize real-world, messy datasets. It takes two massive datasets from the [Austin Open Data Portal](https://data.austintexas.gov/)—a highly fragmented Tree Inventory and a geographically linked 311 Service Requests database—and shows how semantic mapping drastically improves data insights.

## The Problem
The Austin Tree Inventory uses 481 unique, fragmented strings for species names. For example, "Cedar Elm" and "Elm, Cedar" are treated as two distinct species in a raw data query. 311 citizen maintenance requests also have no species column, just geographical coordinates snapped to nearby street intersections. 

## The Solution
This project uses:
1. **Semantic Mapping (OWL/SKOS):** The `austin_trees.ttl` ontology standardizes 481 disparate strings into 101 canonical scientific species.
2. **Spatial Joins (KD-Tree):** Uses `scipy.spatial.cKDTree` to map 311 tickets to the nearest known tree inventory coordinates.

## Project Structure
```text
ReliableAgents_WithOntology/
├── data/                  ← Put your real Austin CSV data here (see instructions below)
├── ontology/              ← austin_trees.ttl (OWL/SKOS mapping)
├── scripts/               
│   ├── evaluate.py        ← The main evaluation pipeline
│   ├── insights.py        ← In-depth analytical inference script
│   └── generate_data.py   ← Synthetic data generator
├── results/               ← output CSVs
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Get the Data

The repository ignores the raw datasets because they are massive (the 311 data is ~1GB). You can download them directly from the Austin Open Data Portal.

1. **Tree Inventory:** [Austin Tree Inventory](https://data.austintexas.gov/Environment/Tree-Inventory/wrik-xasw)
   - Export as CSV and save it as `data/Tree_Inventory_20260801.csv`.
2. **311 Public Data:** [Austin 311 Public Data](https://data.austintexas.gov/Utilities-and-City-Services/Austin-311-Public-Data/xwdj-i9he)
   - Export as CSV and save it as `data/Austin_311_Public_Data_20260801.csv`.

*(Alternatively, run `python3 scripts/generate_data.py` to create small, synthetic sample CSVs in the `data/` folder).*

## Usage

Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Run the Evaluation Pipeline
This matches the 311 tickets to the trees based on proximity, showing the difference in count when using the raw baseline versus the ontology-normalized version.
```bash
python3 scripts/evaluate.py                  # Default 200m radius
python3 scripts/evaluate.py --radius 50      # Strict 50m radius
python3 scripts/evaluate.py --radius 500     # Wide 500m radius
```

### 2. Extract Key Insights
Run the insights script to get deep analyses on tickets-per-tree ratios, emergency vs maintenance patterns, year-over-year trends, and more.
```bash
python3 scripts/insights.py
```

## Sample Output
```text
  ┌─────────────────────────────────────────────────────────────────┐
  │              EVALUATION RESULTS  [radius = 200m]              │
  ├──────────────────────────────┬──────────────┬───────────────────┤
  │ Metric                       │   Baseline   │     Ontology      │
  ├──────────────────────────────┼──────────────┼───────────────────┤
  │ Unique "species" identifiers │          481 │                15 │
  │ Cedar Elm tree count         │        3,979 │             8,041 │
  │ Live Oak tree count          │        5,815 │            11,573 │
  │ Tickets with coordinates     │       48,079 │            48,079 │
  │ Tickets matched (≤200m)      │       13,892 │            13,892 │
  │ Match rate                   │        28.9% │             28.9% │
  │ Live Oak ticket count        │          820 │             2,424 │
  └──────────────────────────────┴──────────────┴───────────────────┘
```
