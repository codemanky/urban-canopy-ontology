import pandas as pd
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)  # Reproducible results

# --- Generate "City Tree Inventory" (Scientific/Formal) ---
# Simulates the Forestry Dept data: precise, scientific names, geolocation.
tree_species = [
    ("Quercus virginiana", "Southern Live Oak"),
    ("Ulmus crassifolia", "Cedar Elm"),
    ("Quercus texana", "Texas Red Oak"),
    ("Platanus occidentalis", "American Sycamore")
]

inventory_data = []
for i in range(1, 101):  # 100 trees
    sci_name, common_name = random.choice(tree_species)
    # 20% chance of missing common name to simulate data gaps
    final_common = common_name if random.random() > 0.2 else None

    inventory_data.append({
        "TREE_ID": f"TR-{1000+i}",
        "SCIENTIFIC_NAME": sci_name,
        "COMMON_NAME": final_common,
        "DBH_INCHES": random.randint(5, 45),
        "LATITUDE": 30.26 + random.uniform(-0.01, 0.01),
        "LONGITUDE": -97.74 + random.uniform(-0.01, 0.01),
        "CONDITION": random.choice(["Good", "Fair", "Critical", "Dead"])
    })

df_trees = pd.DataFrame(inventory_data)
out_trees = DATA_DIR / "trees_sample.csv"
df_trees.to_csv(out_trees, index=False)
print(f"✅ Generated {out_trees.name} (Forestry Dept Data)")

# --- Generate "Maintenance Requests" (Messy/Colloquial) ---
# Simulates 311 citizen reports: vague names, typos, no scientific names.
citizen_terms = {
    "Quercus virginiana": ["Live Oak", "Oak Tree", "Big Oak", "Scrub Oak"],
    "Ulmus crassifolia": ["Elm", "Cedar Elm", "Elm Tree"],
    "Quercus texana": ["Red Oak", "Oak"],
    "Platanus occidentalis": ["Sycamore", "Plane Tree"]
}

maintenance_data = []
for i in range(1, 51):  # 50 requests
    # Pick a random tree from inventory to "link" this request to (implicitly)
    target_tree = random.choice(inventory_data)

    # User uses a vague term (e.g., "Oak" instead of "Quercus virginiana")
    user_term = random.choice(citizen_terms[target_tree["SCIENTIFIC_NAME"]])

    maintenance_data.append({
        "TICKET_ID": f"REQ-{202400+i}",
        "DESCRIPTION": f"Requesting trim for {user_term} blocking sidewalk.",
        "ISSUE_TYPE": random.choice(["Pruning", "Removal", "Health Check"]),
        "REPORTED_SPECIES": user_term,  # The problem column: Non-standard text
        "NEAR_TREE_ID": target_tree["TREE_ID"]  # Ground truth for checking your eval later
    })

df_maint = pd.DataFrame(maintenance_data)
out_maint = DATA_DIR / "maintenance_sample.csv"
df_maint.to_csv(out_maint, index=False)
print(f"✅ Generated {out_maint.name} (Citizen 311 Data)")
