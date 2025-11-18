from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import joblib
from crr_model import risk_category
import math
from datetime import datetime


app = Flask(__name__)
DB_FILE = "card_risk.db"

# Load trained Random Forest model
model_tuple = joblib.load("rf_crr_model.pkl")
model = model_tuple[0]


# -----------------------------
# Helper: Fetch table from DB
# -----------------------------
def fetch_table(table_name, columns):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    col_str = ", ".join(columns)

    query = f"SELECT {col_str} FROM {table_name} ORDER BY ID ASC"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# -----------------------------
# Data access: Individuals & Companies
# -----------------------------
def get_individuals():
    columns = [
        "ID", "Name", "AccountNumber", "ProductType", "CRR_Score", "RiskCategory",
        "Income", "Bankruptcies", "CardLimit", "OpenCreditLines",
        "Age", "EmploymentStatus", "Savings", "CashAdvances", "Last_Updated"
    ]
    return fetch_table("IndividualCardholders", columns)


def get_companies():
    columns = [
        "ID", "Name", "AccountNumber", "ProductType", "CRR_Score", "RiskCategory",
        "Income", "Bankruptcies", "CardLimit", "OpenCreditLines",
        "Age", "EmploymentStatus", "Savings", "CashAdvances", "Role", "Last_Updated"
    ]
    return fetch_table("CompanyCardholders", columns)


# -----------------------------
# Dashboard Route
# -----------------------------
@app.route("/")
def dashboard():
    individuals = get_individuals()
    companies = get_companies()

    def summarize(data):
        high = [d for d in data if str(d["RiskCategory"]).strip().lower() == "high risk"]
        medium = [d for d in data if str(d["RiskCategory"]).strip().lower() == "medium risk"]
        low = [d for d in data if str(d["RiskCategory"]).strip().lower() == "low risk"]
        return [len(high), len(medium), len(low)]

    ind_counts = summarize(individuals)
    comp_counts = summarize(companies)

    # Example placeholders
    predictions = [
        {"customer_id": 'CUST-1001', "credit_limit": 5000},
        {"customer_id": 'CUST-1002', "credit_limit": 12000},
        {"customer_id": 'ENT-5001', "credit_limit": 8000},
        {"customer_id": 'ENT-5002', "credit_limit": 10000}
    ]

    feature_labels = ['Income', 'Credit Utilization', 'Payment History', 'Age', 'Employment Length']
    feature_values = [0.35, 0.25, 0.20, 0.10, 0.10]
    features = list(zip(feature_labels, feature_values))

    return render_template(
        "index.html",
        full_individuals=individuals,
        full_companies=companies,
        ind_counts=ind_counts,
        comp_counts=comp_counts,
        predictions=predictions,
        features=features,
        feature_labels=feature_labels,
        feature_values=feature_values,
        active="dashboard"
    )


# -----------------------------
# Other Routes
# -----------------------------
@app.route("/individuals")
def individuals_page():
    # Always fetch fresh data from DB
    rows = get_individuals()

    # For pie chart counts
    high = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "high risk"]
    medium = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "medium risk"]
    low = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "low risk"]
    ind_counts = [len(high), len(medium), len(low)]

    return render_template(
        "individuals.html",
        rows=rows,
        ind_counts=ind_counts,
        active="individuals"
    )


@app.route("/companies")
def companies_page():
    # Always fetch fresh data from DB
    rows = get_companies()

    # For pie chart counts
    high = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "high risk"]
    medium = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "medium risk"]
    low = [d for d in rows if str(d["RiskCategory"]).strip().lower() == "low risk"]
    comp_counts = [len(high), len(medium), len(low)]

    return render_template(
        "companies.html",
        rows=rows,
        comp_counts=comp_counts,
        active="companies"
    )


@app.route("/heatmaps")
def heatmaps_page():
    return render_template("heatmaps.html", active="heatmaps")


@app.route("/ml")
def ml_page():
    return render_template("ml.html", active="ml")


@app.route("/feature_imp")
def feature_imp_page():
    return render_template("feature_imp.html", active="feature_imp")


# -----------------------------
# Edit Details Page
# -----------------------------
@app.route("/edit_details")
def edit_details_page():
    return render_template("edit_details.html", active="edit_details")


@app.route("/get_user_details", methods=["POST"])
def get_user_details():
    data = request.get_json()
    record_id = data.get("id")
    table_name = data.get("table")

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name} WHERE ID=?", (record_id,))
    rows = cursor.fetchall()
    conn.close()

    # Convert to list of dicts and fix NaN -> None
    rows_list = []
    for r in rows:
        row_dict = dict(r)
        for k, v in row_dict.items():
            if isinstance(v, float) and math.isnan(v):
                row_dict[k] = None
        rows_list.append(row_dict)

    return jsonify({"rows": rows_list})


@app.route("/update_user_details", methods=["POST"])
def update_user_details():
    data = request.get_json()
    record_id = data["id"]
    table_name = data["table"]
    updated_rows = data["rows"]

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Only these columns can be edited
    editable = {
        "Income", "Bankruptcies", "ProductType", "OpenCreditLines",
        "RepaymentHistory", "Savings", "CardLimit", "CashAdvances",
        "Age", "Location", "Citizenship", "Education",
        "EmploymentStatus", "MaritalStatus"
    }

    # Apply updates
    today = datetime.today().strftime("%Y-%m-%d")

    for row in updated_rows:
        row_id = row.get("ID")

        for col, val in row.items():
            if col in editable:
                cursor.execute(
                    f"UPDATE {table_name} SET {col}=? WHERE ID=?",
                    (val, row_id)
                )

        # Always auto-update Last_Updated when any field is edited
        cursor.execute(
            f"UPDATE {table_name} SET Last_Updated=? WHERE ID=?",
            (today, row_id)
        )

    conn.commit()

    # Re-fetch updated row
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE ID=?",
        conn,
        params=(record_id,)
    )

    # ML prediction features
    exclude_cols = ["CRR_Score", "RiskCategory", "ID", "Name", "AccountNumber", "ProductType"]
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    df_features = pd.get_dummies(df_features, drop_first=False)

    feature_names = model_tuple[1]
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0

    df_features = df_features[feature_names]

    # Predict
    preds = model.predict(df_features)
    df["RiskCategory"] = [risk_category(x) for x in preds]

    # Update in DB
    for _, row in df.iterrows():
        cursor.execute(
            f"UPDATE {table_name} SET RiskCategory=? WHERE ID=?",
            (row["RiskCategory"], row["ID"])
        )

    conn.commit()
    conn.close()

    return jsonify({"rows": df.to_dict(orient="records")})

@app.route("/update_row", methods=["POST"])
def update_row():
    data = request.get_json()
    record_id = data["id"]
    table_name = data.get("table", "IndividualCardholders")  # default
    column = data["column"]
    value = data["value"]

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Update ONE CELL
    cursor.execute(
        f"UPDATE {table_name} SET {column}=? WHERE ID=?",
        (value, record_id)
    )
    conn.commit()

    # Recalculate new CRR + Risk based on updated row
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE ID=?",
        conn,
        params=(record_id,)
    )

    # --- preprocessing (same as your update_user_details route) ---
    exclude_cols = ["CRR_Score", "RiskCategory", "ID", "Name", "AccountNumber", "ProductType"]
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    df_features = pd.get_dummies(df_features, drop_first=False)

    feature_names = model_tuple[1]
    missing_cols = [col for col in feature_names if col not in df_features.columns]
    for col in missing_cols:
        df_features[col] = 0
    df_features = df_features[feature_names]

    preds = model.predict(df_features)
    new_risk = risk_category(preds[0])

    # update DB with new risk category
    cursor.execute(
        f"UPDATE {table_name} SET RiskCategory=? WHERE ID=?",
        (new_risk, record_id)
    )
    conn.commit()
    conn.close()

    return jsonify({"new_risk_category": new_risk})


@app.route("/get_individuals_json")
def get_individuals_json():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Select all columns, but we will reorder in Python so ID comes first
    cursor.execute("""
        SELECT AccountNumber, Age, Bankruptcies, CardLimit, CashAdvances,
               Citizenship, CRR_Score, Education, EmploymentStatus,
               Hybrid_CRR, Hybrid_Risk, ID, Income, Last_Updated, Location,
               MaritalStatus, ML_CRR_Pred, Name, OpenCreditLines, ProductType,
               RepaymentHistory, RiskCategory, Savings
        FROM IndividualCardholders
    """)
    rows = cursor.fetchall()
    conn.close()

    # Convert to list of dicts and reorder so ID is first
    data = []
    for r in rows:
        row_dict = dict(r)
        # pop ID and insert at beginning
        id_val = row_dict.pop("ID")
        ordered_row = {"ID": id_val, **row_dict}
        data.append(ordered_row)

    return jsonify({"data": data})

@app.route("/get_companies_json")
def get_companies_json():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Select all columns, but we will reorder in Python so ID comes first
    cursor.execute("""
        SELECT AccountNumber, Age, Bankruptcies, CardLimit, CashAdvances,
               Citizenship, CRR_Score, Education, EmploymentStatus,
               Hybrid_CRR, Hybrid_Risk, ID, Income, Last_Updated, Location,
               MaritalStatus, ML_CRR_Pred, Name, OpenCreditLines, ProductType,
               RepaymentHistory, RiskCategory, Savings, Role
        FROM CompanyCardholders
    """)
    rows = cursor.fetchall()
    conn.close()

    # Convert to list of dicts and reorder so ID is first
    data = []
    for r in rows:
        row_dict = dict(r)
        # pop ID and insert at beginning
        id_val = row_dict.pop("ID")
        ordered_row = {"ID": id_val, **row_dict}
        data.append(ordered_row)

    return jsonify({"data": data})


# Update a row in the DB
@app.route("/update_individual_row", methods=["POST"])
def update_individual_row():
    data = request.get_json()
    record_id = data.get("id")
    column = data.get("column")
    value = data.get("value")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Update the field
    cursor.execute(f"UPDATE IndividualCardholders SET {column}=? WHERE ID=?", (value, record_id))
    conn.commit()

    # Recalculate RiskCategory if needed (replace with your model logic)
    df = pd.read_sql_query("SELECT * FROM IndividualCardholders WHERE ID=?", conn, params=(record_id,))

    # Example: model prediction
    exclude_cols = ["CRR_Score", "RiskCategory", "ID", "Name", "AccountNumber", "ProductType"]
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    df_features = pd.get_dummies(df_features, drop_first=False)
    feature_names = model_tuple[1]
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[feature_names]

    preds = model.predict(df_features)
    df["RiskCategory"] = [risk_category(x) for x in preds]

    # Update RiskCategory in DB
    for index, row in df.iterrows():
        cursor.execute("UPDATE IndividualCardholders SET RiskCategory=? WHERE ID=?", (row["RiskCategory"], row["ID"]))
    conn.commit()
    conn.close()

    # Return updated RiskCategory and optional pie chart counts
    risk_counts = [
        int(pd.Series(df["RiskCategory"]).value_counts().get("High Risk", 0)),
        int(pd.Series(df["RiskCategory"]).value_counts().get("Medium Risk", 0)),
        int(pd.Series(df["RiskCategory"]).value_counts().get("Low Risk", 0)),
    ]

    return jsonify({
        "new_risk_category": df["RiskCategory"].iloc[0],
        "risk_counts": risk_counts
    })


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
