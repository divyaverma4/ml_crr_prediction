import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# SCORING FUNCTIONS  (unchanged)
# -------------------------------
def score_income(income):
    try: inc = float(income)
    except: return 0
    if inc < 40000: return 0
    if inc < 80000: return 5
    if inc < 150000: return 10
    return 15

def score_employment(status):
    if pd.isna(status): return 0
    s = str(status).lower()
    if "unemployed" in s: return 0
    if "self" in s or "part" in s: return 5
    if "full" in s: return 10
    return 5

def score_on_time(repayment):
    if pd.isna(repayment): return 0
    try: val = float(repayment)
    except: return 0
    if val < 70: return 0
    if val <= 90: return 7
    return 15

def score_bankruptcies(val):
    if pd.isna(val): return 0
    return 0 if val > 0 else 8

def score_open_lines(val):
    try: val = int(val)
    except: return 0
    if val > 10: return 0
    if val >= 5: return 3
    return 7

def score_card_limit(val):
    try: val = int(val)
    except: return 0
    return 5

def score_age(age):
    try: age = int(age)
    except: return 0
    if age < 21: return 0
    if age <= 60: return 3
    return 2

def score_citizenship(cit):
    if pd.isna(cit): return 0
    s = str(cit).lower()
    if "non" in s or "temp" in s: return 0
    return 2

def score_savings(val):
    try: val = float(val)
    except: return 0
    if val < 5000: return 0
    if val < 50000: return 5
    return 10

def score_cash_advances(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if "frequent" in s: return 0
    if "occasional" in s: return 3
    return 5

def score_product_type(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if "platinum" in s: return 0
    if "gold" in s: return 3
    return 5

def score_education(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if "high" in s: return 0
    if "college" in s: return 3
    if "grad" in s: return 5
    return 3

def score_region(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if "high" in s: return 0
    if "moderate" in s: return 3
    return 5

def score_marital(val):
    if pd.isna(val): return 1
    s = str(val).lower()
    if "single" in s or "depend" in s: return 1
    return 3

def score_identity_verified(val): return 2

# -------------------------------
# CRR SCORING + CATEGORY
# -------------------------------
def compute_crr(row):
    high = (
        score_income(row.get("Income"))
        + score_employment(row.get("EmploymentStatus"))
        + score_on_time(row.get("RepaymentHistory"))
        + score_bankruptcies(row.get("Bankruptcies"))
        + score_open_lines(row.get("OpenCreditLines"))
        + score_card_limit(row.get("CardLimit"))
        + score_age(row.get("Age"))
        + score_citizenship(row.get("Citizenship"))
    )

    medium = (
        score_savings(row.get("Savings"))
        + score_cash_advances(row.get("CashAdvances"))
        + score_product_type(row.get("ProductType"))
        + score_education(row.get("Education"))
        + score_region(row.get("Location"))
    )

    low = (
        score_marital(row.get("MaritalStatus"))
        + score_identity_verified(row.get("Name"))
    )

    return high + medium + low


def risk_category(score):
    if score >= 65: return "Low Risk"
    if score >= 40: return "Medium Risk"
    return "High Risk"


# -----------------------------------------------------
# ✅ NEW: MISSINGNESS HEATMAP (with CRR classification)
# -----------------------------------------------------
def plot_missingness_heatmap(df, table_name):
    df = df.copy()

    # Ensure risk category has consistent ordering
    df["RiskCategory"] = pd.Categorical(
        df["RiskCategory"],
        categories=["High Risk", "Medium Risk", "Low Risk"],
        ordered=True
    )

    # Sort rows by risk level
    df_sorted = df.sort_values("RiskCategory")

    # Convert missing → 1, present → 0
    missing_matrix = df_sorted.isna().astype(int)

    # Red = missing, Green = present
    cmap = sns.color_palette(["#2ECC71", "#E74C3C"], as_cmap=True)

    plt.figure(figsize=(18, 9))
    sns.heatmap(
        missing_matrix,
        cmap=cmap,
        cbar=True,
        cbar_kws={
            "ticks": [0.25, 0.75],
            "label": "Missingness (0=Present, 1=Missing)"
        }
    )

    plt.title(f"Missingness Heatmap by Risk Category — {table_name}", fontsize=16)
    plt.xlabel("Features")
    plt.ylabel("Records\n(Grouped by Risk Category)")

    # Draw horizontal boundaries between risk groups
    group_sizes = df_sorted["RiskCategory"].value_counts(sort=False)
    boundaries = group_sizes.cumsum().values

    for b in boundaries:
        plt.axhline(b, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{table_name}_missingness_heatmap.png")
    plt.close()

    print(f"📊 Saved heatmap: {table_name}_missingness_heatmap.png")


# -----------------------------------------------------
# PROCESSING FUNCTION
# -----------------------------------------------------
def process_table(db_file, table_name, id_col="ID"):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)

    # Compute CRR + category
    df["CRR_Score"] = df.apply(compute_crr, axis=1)
    df["RiskCategory"] = df["CRR_Score"].apply(risk_category)

    # Replace table
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Updated '{table_name}' with CRR scores and risk categories.\n")

    # Display category counts
    dist = df["RiskCategory"].value_counts().reindex(
        ["High Risk", "Medium Risk", "Low Risk"], fill_value=0
    )
    total = dist.sum()

    print("📊 Risk Category Distribution:")
    for cat, count in dist.items():
        pct = (count / total) * 100 if total else 0
        print(f"   {cat:<12} : {count:>5} ({pct:.1f}%)")

    print()

    # ✅ Generate heatmap
    plot_missingness_heatmap(df, table_name)



# -----------------------------------------------------
# EXECUTION
# -----------------------------------------------------
if __name__ == "__main__":
    db_file = "card_risk.db"
    process_table(db_file, "IndividualCardholders")
    process_table(db_file, "CompanyCardholders")
