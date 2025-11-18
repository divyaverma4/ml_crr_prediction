import sqlite3
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from rf_model import compute_crr, risk_category

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "card_risk.db")
HEATMAP_DIR = os.path.join(PROJECT_ROOT, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ----------------------------
# MISSINGNESS LOGIC
# ----------------------------
def introduce_missingness(row, risk_type):
    """Randomly set fields to NaN based on risk level + missingness tier."""
    base_prob = {"High": 0.25, "Medium": 0.15, "Low": 0.05}
    p = base_prob[risk_type]

    tier = random.choices(
        ["Low", "Medium", "High"],
        weights=[0.6, 0.3, 0.1],
        k=1
    )[0]

    tier_multiplier = {"Low": 1.0, "Medium": 1.8, "High": 3.0}
    p *= tier_multiplier[tier]
    p = min(p, 0.8)  # cap at 80% missing values

    for key in row.keys():
        if key in ["ID", "AccountNumber", "Name"]:
            continue
        if random.random() < p:
            row[key] = np.nan

    return row

# ----------------------------
# RANDOM INDIVIDUAL GENERATOR
# ----------------------------
def random_individual(risk_type):
    row = {}
    if risk_type == "High":
        row.update({
            "Income": random.uniform(1000, 40000),
            "RepaymentHistory": random.uniform(0.2, 0.6),
            "Bankruptcies": random.randint(1, 5),
            "OpenCreditLines": random.randint(1, 3),
            "CardLimit": random.uniform(500, 3000),
            "Age": random.randint(18, 35),
            "Citizenship": "Non-Citizen",
            "Savings": random.uniform(0, 5000),
            "CashAdvances": random.uniform(500, 5000),
            "EmploymentStatus": random.choice(["Unemployed", "Part-Time"]),
            "Education": random.choice(["High School", "None"]),
            "MaritalStatus": random.choice(["Single", "Divorced"]),
            "ProductType": random.choice(["Basic", "Standard"]),
            "Location": random.choice(["Urban", "Rural"]),
        })
    elif risk_type == "Medium":
        row.update({
            "Income": random.uniform(40000, 100000),
            "RepaymentHistory": random.uniform(0.6, 0.9),
            "Bankruptcies": random.randint(0, 1),
            "OpenCreditLines": random.randint(3, 7),
            "CardLimit": random.uniform(3000, 10000),
            "Age": random.randint(25, 50),
            "Citizenship": "Citizen",
            "Savings": random.uniform(5000, 50000),
            "CashAdvances": random.uniform(100, 2000),
            "EmploymentStatus": "Employed",
            "Education": random.choice(["Bachelor", "Master"]),
            "MaritalStatus": random.choice(["Single", "Married"]),
            "ProductType": random.choice(["Standard", "Premium"]),
            "Location": random.choice(["Urban", "Suburban"]),
        })
    else:  # Low risk
        row.update({
            "Income": random.uniform(150000, 250000),
            "RepaymentHistory": random.uniform(0.95, 1.0),
            "Bankruptcies": 0,
            "OpenCreditLines": random.randint(8, 12),
            "CardLimit": random.uniform(15000, 25000),
            "Age": random.randint(30, 60),
            "Citizenship": "Citizen",
            "Savings": random.uniform(50000, 150000),
            "CashAdvances": random.uniform(0, 500),
            "EmploymentStatus": "Employed",
            "Education": "Master",
            "MaritalStatus": "Married",
            "ProductType": "Premium",
            "Location": random.choice(["Urban", "Suburban"]),
        })
    # Identifiers
    row["Name"] = "User" + str(random.randint(1, 999999))
    row["ID"] = random.randint(1000, 9999)
    row["AccountNumber"] = random.randint(100000, 999999)

    row = introduce_missingness(row, risk_type)
    return row

# ----------------------------
# HEATMAP FUNCTION
# ----------------------------
def plot_missingness_heatmap(df, name):
    df_sorted = df.sort_values("RiskCategory")
    missing_matrix = df_sorted.isna().astype(int)
    cmap = sns.color_palette(["#3CB371", "#FF4C4C"], as_cmap=True)

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        missing_matrix,
        cmap=cmap,
        cbar=True,
        cbar_kws={"ticks": [0.25, 0.75], "label": "Missingness (0=Present,1=Missing)"}
    )

    change_points = df_sorted.groupby("RiskCategory").size().cumsum().values
    for cp in change_points:
        plt.axhline(cp, color="black", linewidth=0.8)

    plt.title(f"Missingness Heatmap - {name}", fontsize=14)
    plt.xlabel("Features")
    plt.ylabel("Records")
    plt.tight_layout()

    heatmap_file = os.path.join(HEATMAP_DIR, f"{name}_missingness_heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()
    print(f"📊 Saved missingness heatmap: {heatmap_file}")

# ----------------------------
# GENERATE TABLE
# ----------------------------
def generate_table(n, table_name):
    data = []
    dummy_entries = [
        {
            "Name": "TestLowRisk", "ID": 1111, "AccountNumber": 123456,
            "Income": 200000, "RepaymentHistory": 0.98, "Bankruptcies": 0,
            "OpenCreditLines": 10, "CardLimit": 20000, "Age": 40,
            "Citizenship": "Citizen", "Savings": 100000, "CashAdvances": 0,
            "EmploymentStatus": "Employed", "Education": "Master",
            "MaritalStatus": "Married", "ProductType": "Premium",
            "Location": "Urban"
        },
        {
            "Name": "TestHighRisk", "ID": 2222, "AccountNumber": 234567,
            "Income": 1500, "RepaymentHistory": 0.3, "Bankruptcies": 3,
            "OpenCreditLines": 2, "CardLimit": 1000, "Age": 22,
            "Citizenship": "Non-Citizen", "Savings": 500, "CashAdvances": 2000,
            "EmploymentStatus": "Unemployed", "Education": "High School",
            "MaritalStatus": "Single", "ProductType": "Basic", "Location": "Rural"
        }
    ]
    data.extend(dummy_entries)

    n_low = max(5, int(0.15 * n))
    n_high = max(5, int(0.35 * n))
    n_medium = n - n_low - n_high - len(dummy_entries)

    for _ in range(n_low):
        data.append(random_individual("Low"))
    for _ in range(n_medium):
        data.append(random_individual("Medium"))
    for _ in range(n_high):
        data.append(random_individual("High"))

    df = pd.DataFrame(data)
    df["CRR_Score"] = df.apply(compute_crr, axis=1)
    df["RiskCategory"] = df["CRR_Score"].apply(risk_category)
    df["RiskCategory"] = pd.Categorical(df["RiskCategory"], categories=["Low","Medium","High"], ordered=True)

    print(f"✅ Generated {table_name}: {len(df)} rows")
    print("Risk counts:", df["RiskCategory"].value_counts().to_dict())

    plot_missingness_heatmap(df, table_name)
    return df

# ----------------------------
# MAIN EXECUTION
# ----------------------------
def generate_synthetic_data(n_individuals=10000, n_companies=10000):
    df_individuals = generate_table(n_individuals, "IndividualCardholders")
    df_companies = generate_table(n_companies, "CompanyCardholders")

    conn = sqlite3.connect(DB_PATH)
    df_individuals.to_sql("IndividualCardholders", conn, if_exists="replace", index=False)
    df_companies.to_sql("CompanyCardholders", conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Synthetic CRR database created: {DB_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
