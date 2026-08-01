"""
evaluate.py — Austin Tree Ontology Evaluation Pipeline
=======================================================
Demonstrates the value of the austin_trees.ttl OWL/SKOS ontology by running
a side-by-side comparison of:

  BASELINE  : Raw string matching on the SPECIES column in the tree inventory.
              Fails silently because the same species has multiple spellings.

  ONTOLOGY  : SPARQL-augmented lookup via rdflib maps every SPECIES variant to
              a canonical scientific name, then counts / joins correctly.

Data sources (Austin Open Data Portal):
  - Tree_Inventory_20260801.csv   (62,274 trees)
  - Austin_311_Public_Data_20260801.csv  (2.5M rows → ~48k Tree Issue tickets)

Usage:
  source .venv/bin/activate
  python3 evaluate.py                   # default radius = 200m
  python3 evaluate.py --radius 50       # 50m strict match
  python3 evaluate.py --radius 500      # 500m wide match
  python3 evaluate.py --no-sweep        # skip the radius sweep table
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
from rdflib.namespace import SKOS, RDF
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
TREE_CSV     = BASE / "data" / "Tree_Inventory_20260801.csv"
SR_CSV       = BASE / "data" / "Austin_311_Public_Data_20260801.csv"
ONTOLOGY_TTL = BASE / "ontology" / "austin_trees.ttl"
OUT_CSV      = BASE / "results" / "evaluation_results.csv"

# ---------------------------------------------------------------------------
# CLI Arguments — radius is configurable, everything else has sensible defaults
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Austin Tree Ontology Evaluation Pipeline"
)
parser.add_argument(
    "--radius",
    type=float,
    default=200.0,
    metavar="METRES",
    help="Spatial match radius in metres (default: 200m). "
         "311 coordinates snap to street intersections so ≥100m is recommended.",
)
parser.add_argument(
    "--no-sweep",
    action="store_true",
    default=False,
    help="Skip the multi-radius sweep table (faster for large datasets).",
)
args = parser.parse_args()

RADIUS_M   = args.radius
RADIUS_DEG = RADIUS_M / 111_000          # 1 degree lat ≈ 111km everywhere
DO_SWEEP   = not args.no_sweep

# Sweep radii to benchmark (metres)
SWEEP_RADII_M = [50, 100, 200, 500, 1000]

# ---------------------------------------------------------------------------
# Ontology namespaces
# ---------------------------------------------------------------------------
DWC   = Namespace("http://rs.tdwg.org/dwc/terms/")
ATREE = Namespace("http://austin.gov/ontology/trees#")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str):
    print()
    print(f"── {title} " + "─" * max(0, 65 - len(title)))


def fmt(n: int) -> str:
    return f"{n:,}"


# ===========================================================================
# STEP 1 — Load the Tree Inventory
# ===========================================================================
banner("STEP 1 — Loading Austin Tree Inventory")

print(f"  Reading {TREE_CSV.name} …", end=" ", flush=True)
t0 = time.time()
trees = pd.read_csv(TREE_CSV, low_memory=False)
print(f"done ({time.time()-t0:.1f}s)  →  {fmt(len(trees))} rows")

# Normalise column names
trees.columns = [c.strip() for c in trees.columns]
trees["LATITUDE"]   = pd.to_numeric(trees["LATITUDE"],   errors="coerce")
trees["LONGTITUDE"] = pd.to_numeric(trees["LONGTITUDE"], errors="coerce")
trees["DIAMETER"]   = pd.to_numeric(trees["DIAMETER"],   errors="coerce")
trees["SPECIES"]    = trees["SPECIES"].fillna("Unknown").str.strip()

# Drop rows with no location (can't do spatial join)
trees_geo = trees.dropna(subset=["LATITUDE", "LONGTITUDE"]).copy()
trees_geo = trees_geo[
    (trees_geo["LATITUDE"].between(29.0, 31.5)) &
    (trees_geo["LONGTITUDE"].between(-98.5, -96.5))
].copy()
trees_geo = trees_geo.reset_index(drop=True)

print(f"  Trees with valid coordinates: {fmt(len(trees_geo))}")
print(f"  Unique SPECIES strings in inventory: {trees['SPECIES'].nunique()}")

# ===========================================================================
# STEP 2 — Load the OWL Ontology
# ===========================================================================
banner("STEP 2 — Loading OWL/SKOS Ontology")

print(f"  Parsing {ONTOLOGY_TTL.name} …", end=" ", flush=True)
g = Graph()
g.parse(ONTOLOGY_TTL, format="turtle")
print(f"done  →  {len(g)} triples")

# Build label → scientific-name lookup from ontology via SPARQL
SPARQL_LOOKUP = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dwc:  <http://rs.tdwg.org/dwc/terms/>

SELECT ?label ?sciName ?prefLabel
WHERE {
    ?species dwc:scientificName ?sciName ;
             skos:prefLabel     ?prefLabel .
    { ?species skos:altLabel  ?label }
    UNION
    { ?species skos:prefLabel ?label }
}
"""

results = g.query(SPARQL_LOOKUP)
label_to_sci   = {}   # raw label (lower) → scientific name
label_to_pref  = {}   # raw label (lower) → preferred common name

for row in results:
    key = str(row.label).strip().lower()
    label_to_sci[key]  = str(row.sciName)
    label_to_pref[key] = str(row.prefLabel)

print(f"  Ontology term mappings loaded: {fmt(len(label_to_sci))} label → scientific-name pairs")

def ontology_lookup(species_str: str):
    """Return (scientific_name, pref_label) for a SPECIES string via ontology."""
    key = species_str.strip().lower()
    return label_to_sci.get(key, None), label_to_pref.get(key, None)


# ===========================================================================
# STEP 3 — BASELINE: Species Counting (Exact String Match)
# ===========================================================================
banner("STEP 3 — BASELINE: Exact String Species Counts")

baseline_counts = trees["SPECIES"].value_counts()

# Focus on species we know are fragmented
FRAGMENTED_SPECIES = {
    "Cedar Elm / Ulmus crassifolia": ["Cedar Elm", "Elm, Cedar"],
    "Live Oak / Quercus virginiana": [
        "Southern Live Oak",
        "Oak, Live (Southern)",
        "Oak, Texas Live (Escarpment)",
        "Escarpment Live Oak",
    ],
    "Crape Myrtle / Lagerstroemia indica": [
        "Crapemyrtle",
        "Crape Myrtle (including hybrids)",
    ],
    "Ashe Juniper / Juniperus ashei": ["Ashe Juniper", "Juniper, Ashe"],
    "Hackberry / Celtis spp.": [
        "Hackberry, Common",
        "Sugarberry",
        "Hackberry",
    ],
}

section("Fragmented species counts (BASELINE — exact string)")
print(f"  {'Species Group':<45} {'Variant String':<40} {'Count':>6}")
print(f"  {'-'*45} {'-'*40} {'-'*6}")
baseline_totals = {}
for group, variants in FRAGMENTED_SPECIES.items():
    group_total = 0
    first = True
    for v in variants:
        cnt = int(baseline_counts.get(v, 0))
        group_total += cnt
        label = group if first else ""
        print(f"  {label:<45} {v:<40} {fmt(cnt):>6}")
        first = False
    baseline_totals[group] = group_total
print()


# ===========================================================================
# STEP 4 — ONTOLOGY: Normalise Every Tree to Canonical Scientific Name
# ===========================================================================
banner("STEP 4 — ONTOLOGY: Normalise SPECIES → Scientific Name")

trees_geo["SCI_NAME"]  = None
trees_geo["PREF_LABEL"] = None

for idx, row in trees_geo.iterrows():
    sci, pref = ontology_lookup(row["SPECIES"])
    trees_geo.at[idx, "SCI_NAME"]   = sci if sci else "Unmapped"
    trees_geo.at[idx, "PREF_LABEL"] = pref if pref else row["SPECIES"]

total = len(trees_geo)
mapped   = (trees_geo["SCI_NAME"] != "Unmapped").sum()
unmapped = total - mapped

section("Ontology normalisation summary")
print(f"  Total trees (with coordinates): {fmt(total)}")
print(f"  Successfully mapped to ontology: {fmt(mapped)}  ({mapped/total*100:.1f}%)")
print(f"  Unmapped (not in ontology):       {fmt(unmapped)}  ({unmapped/total*100:.1f}%)")

section("Fragmented species — BASELINE vs ONTOLOGY counts")
print(f"  {'Species Group':<45} {'Baseline':>9} {'Ontology':>9} {'Gain':>8}")
print(f"  {'-'*45} {'-'*9} {'-'*9} {'-'*8}")

for group, variants in FRAGMENTED_SPECIES.items():
    # Ontology count: look up canonical sci name from first variant
    sci_name, _ = ontology_lookup(variants[0])
    if sci_name:
        onto_cnt = int((trees_geo["SCI_NAME"] == sci_name).sum())
    else:
        onto_cnt = baseline_totals[group]
    base_cnt = baseline_totals[group]
    gain_pct = (onto_cnt - base_cnt) / base_cnt * 100 if base_cnt else 0
    # The baseline "best case" is just the most common single variant
    best_baseline = max(int(trees["SPECIES"].value_counts().get(v, 0)) for v in variants)
    print(f"  {group:<45} {fmt(best_baseline):>9} {fmt(onto_cnt):>9} {gain_pct:>+7.0f}%")

section("Top 15 canonical species in inventory (Ontology view)")
canon_counts = (
    trees_geo[trees_geo["SCI_NAME"] != "Unmapped"]
    .groupby(["SCI_NAME", "PREF_LABEL"])
    .size()
    .reset_index(name="TREE_COUNT")
    .sort_values("TREE_COUNT", ascending=False)
    .head(15)
)
print(f"  {'Rank':<5} {'Scientific Name':<30} {'Common Name':<30} {'Trees':>7}")
print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*7}")
for rank, (_, r) in enumerate(canon_counts.iterrows(), 1):
    print(f"  {rank:<5} {r['SCI_NAME']:<30} {r['PREF_LABEL']:<30} {fmt(r['TREE_COUNT']):>7}")


# ===========================================================================
# STEP 5 — Load & Filter 311 Tree Issue Tickets
# ===========================================================================
banner("STEP 5 — Loading 311 Tree Issue Tickets")

print(f"  Reading {SR_CSV.name} (2.5M rows — may take ~30s) …", end=" ", flush=True)
t0 = time.time()
sr = pd.read_csv(SR_CSV, low_memory=False)
print(f"done ({time.time()-t0:.1f}s)  →  {fmt(len(sr))} total rows")

# Filter to tree issue tickets only
tree_mask = sr["SR Description"].str.contains("Tree Issue", na=False, case=False)
tree_sr = sr[tree_mask].copy()
print(f"  'Tree Issue' tickets: {fmt(len(tree_sr))}")

# Normalise coordinates
tree_sr["LAT"] = pd.to_numeric(tree_sr["Latitude Coordinate"],  errors="coerce")
tree_sr["LON"] = pd.to_numeric(tree_sr["Longitude Coordinate"], errors="coerce")
tree_sr = tree_sr.dropna(subset=["LAT", "LON"]).copy()
tree_sr = tree_sr[
    tree_sr["LAT"].between(29.0, 31.5) &
    tree_sr["LON"].between(-98.5, -96.5)
].copy()
tree_sr = tree_sr.reset_index(drop=True)
print(f"  Tickets with valid coordinates: {fmt(len(tree_sr))}")

section("311 Tree Issue ticket types")
print(tree_sr["SR Description"].value_counts().to_string())


# ===========================================================================
# STEP 6 — Spatial Join: Match Each 311 Ticket to Nearest Tree
# ===========================================================================
banner("STEP 6 — Spatial Join: 311 Tickets → Nearest Tree (KD-Tree)")

# Build KD-tree from tree coordinates (degrees ≈ 111km/deg lat, 95km/deg lon in Austin)
# Scale lon to match lat distances approximately
LAT_SCALE = 1.0
LON_SCALE = np.cos(np.radians(30.26))  # Austin latitude

tree_coords = np.column_stack([
    trees_geo["LATITUDE"].values * LAT_SCALE,
    trees_geo["LONGTITUDE"].values * LON_SCALE,
])
ticket_coords = np.column_stack([
    tree_sr["LAT"].values * LAT_SCALE,
    tree_sr["LON"].values * LON_SCALE,
])

print(f"  Building KD-tree from {fmt(len(trees_geo))} tree locations …", end=" ", flush=True)
t0 = time.time()
kdtree = cKDTree(tree_coords)
print(f"done ({time.time()-t0:.2f}s)")

print(f"  Active radius  : {RADIUS_M:.0f}m  (≈ {RADIUS_DEG:.6f}°)")
print(f"  Tip: re-run with --radius <metres> to change  |  --no-sweep to skip sweep")

print(f"  Querying nearest tree for {fmt(len(tree_sr))} tickets …", end=" ", flush=True)
t0 = time.time()
distances, indices = kdtree.query(ticket_coords, k=1, workers=-1)
print(f"done ({time.time()-t0:.2f}s)")

tree_sr = tree_sr.copy()
tree_sr["NEAREST_TREE_IDX"]  = indices
tree_sr["NEAREST_DIST_DEG"]  = distances
tree_sr["NEAREST_DIST_M"]    = distances * 111_000
tree_sr["WITHIN_RADIUS"]     = distances <= RADIUS_DEG

matched   = int(tree_sr["WITHIN_RADIUS"].sum())
unmatched = len(tree_sr) - matched
print(f"  Tickets matched (≤ {RADIUS_M:.0f}m): {fmt(matched)}  ({matched/len(tree_sr)*100:.1f}%)")
print(f"  Tickets unmatched:          {fmt(unmatched)}  ({unmatched/len(tree_sr)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Radius sweep — shows match rate vs. precision tradeoff across multiple radii
# ---------------------------------------------------------------------------
if DO_SWEEP:
    section("Radius sweep — match rate vs. precision tradeoff")
    print(f"  {'Radius':>8}  {'Matched':>8}  {'Match %':>8}  {'Avg dist (m)':>13}  {'Interpretation'}")
    print(f"  {'-------':>8}  {'-------':>8}  {'-------':>8}  {'-----------':>13}  ---------------")
    for r_m in SWEEP_RADII_M:
        r_deg  = r_m / 111_000
        within = (distances <= r_deg).sum()
        pct    = within / len(distances) * 100
        # avg distance only for matched tickets
        mask   = distances <= r_deg
        avg_m  = (distances[mask] * 111_000).mean() if mask.any() else 0
        note   = (
            "← too strict (intersect snap)" if r_m <= 50
            else "← recommended minimum"    if r_m == 100
            else "← current run"            if r_m == RADIUS_M
            else ""
        )
        marker = " ◀" if r_m == RADIUS_M else ""
        print(f"  {r_m:>6}m   {fmt(within):>8}  {pct:>7.1f}%  {avg_m:>11.1f}m  {note}{marker}")

# Attach tree data to matched tickets
matched_sr = tree_sr[tree_sr["WITHIN_RADIUS"]].copy()
matched_sr["TREE_SPECIES"]    = trees_geo.iloc[matched_sr["NEAREST_TREE_IDX"].values]["SPECIES"].values
matched_sr["TREE_SCI_NAME"]   = trees_geo.iloc[matched_sr["NEAREST_TREE_IDX"].values]["SCI_NAME"].values
matched_sr["TREE_PREF_LABEL"] = trees_geo.iloc[matched_sr["NEAREST_TREE_IDX"].values]["PREF_LABEL"].values
matched_sr["TREE_DIAMETER"]   = trees_geo.iloc[matched_sr["NEAREST_TREE_IDX"].values]["DIAMETER"].values


# ===========================================================================
# STEP 7 — Baseline vs Ontology: Maintenance Hotspot Analysis
# ===========================================================================
banner("STEP 7 — Baseline vs Ontology: Maintenance Hotspot Analysis")

section(f"BASELINE: Tickets per raw SPECIES string (top 15 — fragmented)  [radius={RADIUS_M:.0f}m]")
baseline_hotspot = (
    matched_sr.groupby("TREE_SPECIES")
    .size()
    .reset_index(name="TICKET_COUNT")
    .sort_values("TICKET_COUNT", ascending=False)
    .head(15)
)
print(f"  {'Rank':<5} {'SPECIES (raw)':<45} {'Tickets':>8}")
print(f"  {'-'*5} {'-'*45} {'-'*8}")
for rank, (_, r) in enumerate(baseline_hotspot.iterrows(), 1):
    print(f"  {rank:<5} {r['TREE_SPECIES']:<45} {fmt(r['TICKET_COUNT']):>8}")

section(f"ONTOLOGY: Tickets per canonical species (top 15 — unified)  [radius={RADIUS_M:.0f}m]")
mapped_sr = matched_sr[matched_sr["TREE_SCI_NAME"] != "Unmapped"]
ontology_hotspot = (
    mapped_sr.groupby(["TREE_SCI_NAME", "TREE_PREF_LABEL"])
    .size()
    .reset_index(name="TICKET_COUNT")
    .sort_values("TICKET_COUNT", ascending=False)
    .head(15)
)
print(f"  {'Rank':<5} {'Scientific Name':<30} {'Common Name':<30} {'Tickets':>8}")
print(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*8}")
for rank, (_, r) in enumerate(ontology_hotspot.iterrows(), 1):
    print(f"  {rank:<5} {r['TREE_SCI_NAME']:<30} {r['TREE_PREF_LABEL']:<30} {fmt(r['TICKET_COUNT']):>8}")

section("Key insight: Live Oak undercounting in BASELINE")
oak_variants = ["Southern Live Oak", "Oak, Live (Southern)", "Oak, Texas Live (Escarpment)", "Escarpment Live Oak"]
base_oak = int(baseline_hotspot[baseline_hotspot["TREE_SPECIES"].isin(oak_variants)]["TICKET_COUNT"].sum())
onto_oak_row = ontology_hotspot[ontology_hotspot["TREE_SCI_NAME"] == "Quercus virginiana"]
onto_oak = int(onto_oak_row["TICKET_COUNT"].values[0]) if len(onto_oak_row) else 0
top_base_oak_rows = baseline_hotspot[baseline_hotspot["TREE_SPECIES"] == "Southern Live Oak"]
top_base_oak = int(top_base_oak_rows["TICKET_COUNT"].values[0]) if len(top_base_oak_rows) else 0
missed = onto_oak - top_base_oak
missed_pct = missed / onto_oak * 100 if onto_oak else 0

print(f"""
  Live Oak (Quercus virginiana) maintenance tickets  [radius = {RADIUS_M:.0f}m]:
    Baseline — top single variant only ("Southern Live Oak"):  {fmt(top_base_oak)}
    Baseline — all 4 variants summed manually:                 {fmt(base_oak)}
    Ontology — single canonical query (Quercus virginiana):    {fmt(onto_oak)}

  ⚠  Without the ontology, a query for "Southern Live Oak" misses
     {fmt(missed)} tickets ({missed_pct:.0f}% of the true total).
""")


# ===========================================================================
# STEP 8 — Write Results CSV
# ===========================================================================
banner("STEP 8 — Writing evaluation_results.csv")

out_cols = [
    "Service Request (SR) Number",
    "SR Description",
    "SR Location",
    "LAT",
    "LON",
    "WITHIN_RADIUS",
    "NEAREST_DIST_M",
    "NEAREST_DIST_DEG",
    "TREE_SPECIES",
    "TREE_SCI_NAME",
    "TREE_PREF_LABEL",
    "TREE_DIAMETER",
]
tree_sr_out = tree_sr.copy()
for col in ["TREE_SPECIES", "TREE_SCI_NAME", "TREE_PREF_LABEL", "TREE_DIAMETER"]:
    tree_sr_out[col] = None

# Fill matched rows
tree_sr_out.loc[matched_sr.index, "TREE_SPECIES"]    = matched_sr["TREE_SPECIES"].values
tree_sr_out.loc[matched_sr.index, "TREE_SCI_NAME"]   = matched_sr["TREE_SCI_NAME"].values
tree_sr_out.loc[matched_sr.index, "TREE_PREF_LABEL"] = matched_sr["TREE_PREF_LABEL"].values
tree_sr_out.loc[matched_sr.index, "TREE_DIAMETER"]   = matched_sr["TREE_DIAMETER"].values
tree_sr_out.loc[matched_sr.index, "NEAREST_DIST_M"]  = matched_sr["NEAREST_DIST_M"].values

available_cols = [c for c in out_cols if c in tree_sr_out.columns]
tree_sr_out[available_cols].to_csv(OUT_CSV, index=False)
print(f"  Saved {fmt(len(tree_sr_out))} rows → {OUT_CSV.name}")


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
banner("FINAL SUMMARY — Ontology Value Demonstrated")

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │              EVALUATION RESULTS  [radius = {RADIUS_M:.0f}m]              │
  ├──────────────────────────────┬──────────────┬───────────────────┤
  │ Metric                       │   Baseline   │     Ontology      │
  ├──────────────────────────────┼──────────────┼───────────────────┤
  │ Unique "species" identifiers │ {trees['SPECIES'].nunique():>12} │ {canon_counts['SCI_NAME'].nunique():>17} │
  │ Cedar Elm tree count         │ {fmt(int(baseline_counts.get('Cedar Elm',0))):>12} │ {fmt(int((trees_geo['SCI_NAME']=='Ulmus crassifolia').sum())):>17} │
  │ Live Oak tree count          │ {fmt(int(baseline_counts.get('Southern Live Oak',0))):>12} │ {fmt(int((trees_geo['SCI_NAME']=='Quercus virginiana').sum())):>17} │
  │ Tickets with coordinates     │ {fmt(len(tree_sr)):>12} │ {fmt(len(tree_sr)):>17} │
  │ Tickets matched (≤{RADIUS_M:.0f}m)     │ {fmt(matched):>12} │ {fmt(matched):>17} │
  │ Match rate                   │ {matched/len(tree_sr)*100:>11.1f}% │ {matched/len(tree_sr)*100:>16.1f}% │
  │ Live Oak ticket count        │ {fmt(top_base_oak):>12} │ {fmt(onto_oak):>17} │
  └──────────────────────────────┴──────────────┴───────────────────┘

  Key Findings:
  ① The Austin Tree Inventory uses {trees['SPECIES'].nunique()} unique SPECIES strings for
    what the ontology resolves to {canon_counts['SCI_NAME'].nunique()} canonical species.
  ② A baseline query for "Cedar Elm" finds {fmt(int(baseline_counts.get('Cedar Elm',0)))} trees,
    but the ontology finds {fmt(int((trees_geo['SCI_NAME']=='Ulmus crassifolia').sum()))} — a
    {(int((trees_geo['SCI_NAME']=='Ulmus crassifolia').sum())/int(baseline_counts.get('Cedar Elm',0))-1)*100:.0f}% undercount without the ontology.
  ③ {fmt(matched)} of {fmt(len(tree_sr))} tree issue tickets matched a tree within {RADIUS_M:.0f}m.
    Run with --radius 500 for wider coverage, --radius 50 for strict precision.
  ④ Results saved to evaluation_results.csv for further analysis.
""")
