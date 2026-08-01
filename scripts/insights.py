"""
insights.py — Deep Inference Analysis for Austin Tree Ontology
=============================================================
This script provides 5 deep inferences by joining the Austin 311 
service requests dataset and the Tree Inventory using the OWL/SKOS ontology.

Usage:
  source .venv/bin/activate
  python3 scripts/insights.py               # default radius = 200m
  python3 scripts/insights.py --radius 100  # tighter 100m match
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdflib import Graph
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Austin Tree Ontology Insights")
parser.add_argument(
    "--radius",
    type=float,
    default=200.0,
    metavar="METRES",
    help="Spatial match radius in metres (default: 200m).",
)
args = parser.parse_args()
RADIUS_M = args.radius
RADIUS_DEG = RADIUS_M / 111_000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
TREE_CSV     = BASE / "data" / "Tree_Inventory_20260801.csv"
SR_CSV       = BASE / "data" / "Austin_311_Public_Data_20260801.csv"
ONTOLOGY_TTL = BASE / "ontology" / "austin_trees.ttl"

# ---------------------------------------------------------------------------
# Load Ontology Labels
# ---------------------------------------------------------------------------
print(f"Loading ontology from {ONTOLOGY_TTL.name}...")
g = Graph()
g.parse(ONTOLOGY_TTL, format='turtle')
SPARQL = '''
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dwc:  <http://rs.tdwg.org/dwc/terms/>
SELECT ?label ?sciName ?prefLabel WHERE {
    ?species dwc:scientificName ?sciName ;
             skos:prefLabel     ?prefLabel .
    { ?species skos:altLabel  ?label }
    UNION
    { ?species skos:prefLabel ?label }
}'''
label_to_sci  = {str(r.label).strip().lower(): str(r.sciName)  for r in g.query(SPARQL)}
label_to_pref = {str(r.label).strip().lower(): str(r.prefLabel) for r in g.query(SPARQL)}

# ---------------------------------------------------------------------------
# Load Tree Inventory
# ---------------------------------------------------------------------------
print(f"Loading tree inventory from {TREE_CSV.name}...")
trees = pd.read_csv(TREE_CSV, low_memory=False)
trees.columns = [c.strip() for c in trees.columns]
trees['SPECIES']    = trees['SPECIES'].fillna('Unknown').str.strip()
trees['LATITUDE']   = pd.to_numeric(trees['LATITUDE'],   errors='coerce')
trees['LONGTITUDE'] = pd.to_numeric(trees['LONGTITUDE'], errors='coerce')
trees['DIAMETER']   = pd.to_numeric(trees['DIAMETER'],   errors='coerce')
trees['SCI_NAME']   = trees['SPECIES'].str.strip().str.lower().map(label_to_sci).fillna('Unmapped')
trees['PREF_LABEL'] = trees['SPECIES'].str.strip().str.lower().map(label_to_pref).fillna(trees['SPECIES'])

trees_geo = trees.dropna(subset=['LATITUDE','LONGTITUDE']).copy()
trees_geo = trees_geo[trees_geo['LATITUDE'].between(29,31.5) & trees_geo['LONGTITUDE'].between(-98.5,-96.5)].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load 311 Tickets
# ---------------------------------------------------------------------------
print(f"Loading 311 tickets from {SR_CSV.name}...")
sr = pd.read_csv(SR_CSV, low_memory=False)
tree_sr = sr[sr['SR Description'].str.contains('Tree Issue', na=False, case=False)].copy()
tree_sr['LAT'] = pd.to_numeric(tree_sr['Latitude Coordinate'], errors='coerce')
tree_sr['LON'] = pd.to_numeric(tree_sr['Longitude Coordinate'], errors='coerce')
tree_sr = tree_sr.dropna(subset=['LAT','LON'])
tree_sr = tree_sr[tree_sr['LAT'].between(29,31.5) & tree_sr['LON'].between(-98.5,-96.5)].reset_index(drop=True)
tree_sr['CREATED'] = pd.to_datetime(tree_sr['Created Date'], errors='coerce')
tree_sr['YEAR']    = tree_sr['CREATED'].dt.year
tree_sr['MONTH']   = tree_sr['CREATED'].dt.month

# ---------------------------------------------------------------------------
# Spatial Join
# ---------------------------------------------------------------------------
print(f"Running spatial join at {RADIUS_M}m radius...")
LAT_SCALE = 1.0
LON_SCALE = np.cos(np.radians(30.26))
tree_coords   = np.column_stack([trees_geo['LATITUDE'].values * LAT_SCALE, trees_geo['LONGTITUDE'].values * LON_SCALE])
ticket_coords = np.column_stack([tree_sr['LAT'].values * LAT_SCALE, tree_sr['LON'].values * LON_SCALE])
kdtree = cKDTree(tree_coords)
distances, indices = kdtree.query(ticket_coords, k=1, workers=-1)

tree_sr['WITHIN_RADIUS'] = distances <= RADIUS_DEG
tree_sr['TREE_SCI']      = np.where(tree_sr['WITHIN_RADIUS'], trees_geo['SCI_NAME'].iloc[indices].values,  None)
tree_sr['TREE_PREF']     = np.where(tree_sr['WITHIN_RADIUS'], trees_geo['PREF_LABEL'].iloc[indices].values, None)
tree_sr['TREE_DIAM']     = np.where(tree_sr['WITHIN_RADIUS'], trees_geo['DIAMETER'].iloc[indices].values,   np.nan)

matched = tree_sr[tree_sr['WITHIN_RADIUS'] & (tree_sr['TREE_SCI'] != 'Unmapped')].copy()

print(f"Matched tickets: {len(matched):,}\n")

# ===========================================================================
# INSIGHT 1: Tickets-per-tree (complaint rate by species)
# ===========================================================================
print("="*65)
print("INSIGHT 1: Tickets-per-tree (complaint rate by species)")
print("="*65)
ticket_by_sci = matched.groupby('TREE_SCI').size().reset_index(name='TICKETS')
tree_by_sci   = trees_geo[trees_geo['SCI_NAME']!='Unmapped'].groupby('SCI_NAME').size().reset_index(name='TREE_COUNT')
merged = ticket_by_sci.merge(tree_by_sci, left_on='TREE_SCI', right_on='SCI_NAME')
merged['TICKETS_PER_100_TREES'] = merged['TICKETS'] / merged['TREE_COUNT'] * 100
merged = merged.merge(
    trees_geo[['SCI_NAME','PREF_LABEL']].drop_duplicates(),
    left_on='TREE_SCI', right_on='SCI_NAME', how='left'
).sort_values('TICKETS_PER_100_TREES', ascending=False)

print(f"  {'Species':<30} {'Trees':>7} {'Tickets':>8} {'Per 100 trees':>14}")
print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*14}")
for _, r in merged.head(15).iterrows():
    label = str(r.get('PREF_LABEL', r['TREE_SCI']))[:30]
    print(f"  {label:<30} {r['TREE_COUNT']:>7,} {r['TICKETS']:>8,} {r['TICKETS_PER_100_TREES']:>13.1f}%")

# ===========================================================================
# INSIGHT 2: Emergency vs Maintenance by species
# ===========================================================================
print("\n" + "="*65)
print("INSIGHT 2: Emergency vs Maintenance ticket split by species")
print("="*65)
matched['IS_EMERGENCY'] = matched['SR Description'].str.contains('Emergency', case=False, na=False)
emerg_by_sci = matched.groupby('TREE_SCI').agg(
    TICKETS=('TREE_SCI','count'),
    EMERGENCY=('IS_EMERGENCY','sum')
).reset_index()
emerg_by_sci['EMERG_PCT'] = emerg_by_sci['EMERGENCY'] / emerg_by_sci['TICKETS'] * 100
emerg_by_sci = emerg_by_sci.merge(
    trees_geo[['SCI_NAME','PREF_LABEL']].drop_duplicates(), left_on='TREE_SCI', right_on='SCI_NAME', how='left'
).sort_values('EMERG_PCT', ascending=False)
print(f"  {'Species':<30} {'Tickets':>8} {'Emergency':>10} {'Emerg %':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
for _, r in emerg_by_sci[emerg_by_sci['TICKETS']>=20].head(12).iterrows():
    label = str(r.get('PREF_LABEL', r['TREE_SCI']))[:30]
    print(f"  {label:<30} {r['TICKETS']:>8,} {int(r['EMERGENCY']):>10,} {r['EMERG_PCT']:>7.1f}%")

# ===========================================================================
# INSIGHT 3: Diameter (size) of ticketed trees vs all trees
# ===========================================================================
print("\n" + "="*65)
print("INSIGHT 3: Do larger trees generate more complaints?")
print("="*65)
all_diam    = trees_geo['DIAMETER'].dropna()
ticket_diam = matched['TREE_DIAM'].dropna().astype(float)
print(f"  Avg diameter — ALL trees in inventory : {all_diam.mean():.1f} inches")
print(f"  Avg diameter — Trees with 311 tickets : {ticket_diam.mean():.1f} inches")
print(f"  Median diam  — ALL trees              : {all_diam.median():.1f} inches")
print(f"  Median diam  — Ticketed trees         : {ticket_diam.median():.1f} inches\n")

bins = [0,6,12,24,36,200]
labels = ['0-6"','6-12"','12-24"','24-36"','36"+']
all_cut    = pd.cut(all_diam, bins=bins, labels=labels)
ticket_cut = pd.cut(ticket_diam, bins=bins, labels=labels)
all_pct    = all_cut.value_counts(normalize=True).sort_index() * 100
ticket_pct = ticket_cut.value_counts(normalize=True).sort_index() * 100
print(f"  {'Diameter Band':<12} {'All trees %':>12} {'Ticketed %':>12} {'Δ':>6}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
for band in labels:
    a = all_pct.get(band, 0)
    t = ticket_pct.get(band, 0)
    print(f"  {band:<12} {a:>11.1f}%  {t:>11.1f}%  {t-a:>+5.1f}%")

# ===========================================================================
# INSIGHT 4: Year-over-year ticket trend
# ===========================================================================
print("\n" + "="*65)
print("INSIGHT 4: Year-over-year ticket trend (all tree issues)")
print("="*65)
yoy = tree_sr.groupby('YEAR').size().reset_index(name='TICKETS')
yoy = yoy[yoy['YEAR'].between(2014, 2026)]
yoy['CHANGE'] = yoy['TICKETS'].pct_change() * 100
print(f"  {'Year':<6} {'Tickets':>8} {'YoY Change':>12}")
print(f"  {'-'*6} {'-'*8} {'-'*12}")
for _, r in yoy.iterrows():
    chg = f"{r['CHANGE']:+.1f}%" if pd.notna(r['CHANGE']) else "  --"
    print(f"  {int(r['YEAR']):<6} {int(r['TICKETS']):>8,} {chg:>12}")

# ===========================================================================
# INSIGHT 5: Seasonal pattern
# ===========================================================================
print("\n" + "="*65)
print("INSIGHT 5: Seasonal pattern (all years combined)")
print("="*65)
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasonal = tree_sr.groupby('MONTH').size().reset_index(name='TICKETS')
total_tickets = seasonal['TICKETS'].sum()
for _, r in seasonal.iterrows():
    m = int(r['MONTH'])
    bar = '█' * int(r['TICKETS'] / total_tickets * 240)
    print(f"  {MONTH_NAMES[m-1]:<4} {int(r['TICKETS']):>6,}  {bar}")
