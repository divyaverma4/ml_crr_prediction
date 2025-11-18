import sqlite3
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from crr_model import compute_crr, risk_category

# ---------- 1. LOAD DATA ----------
def load_data(db_file, table_name):
    conn = sqlite3.connect(db_file)
    df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    conn.close()
    return df

# ---------- 2. PREPROCESS ----------
def preprocess_data(df):
    df = df.copy()
    if "CRR_Score" not in df.columns:
        df["CRR_Score"] = df.apply(compute_crr, axis=1)
        df["RiskCategory"] = df["CRR_Score"].apply(risk_category)

    exclude_cols = [
        "CRR_Score", "RiskCategory", "ML_CRR_Pred", "Hybrid_CRR", "Hybrid_Risk",
        "ID", "Name", "AccountNumber"
    ]
    df_features = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    num_cols = df_features.select_dtypes(include="number").columns
    cat_cols = df_features.select_dtypes(exclude="number").columns

    num_imputer = SimpleImputer(strategy="median")
    df_features[num_cols] = num_imputer.fit_transform(df_features[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_features[cat_cols] = cat_imputer.fit_transform(df_features[cat_cols])

    df_features = pd.get_dummies(df_features, drop_first=False)
    df_features["CRR_Score"] = df["CRR_Score"]

    return df_features

# ---------- 3. SAVE MISSINGNESS HEATMAP ----------
def save_missingness_heatmap(df, table_name):
    os.makedirs("static/images", exist_ok=True)
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(f"{table_name} Missingness Heatmap")
    file_path = f"static/images/{table_name}_missingness_heatmap.png"
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"📊 Saved heatmap: {file_path}")

# ---------- 4. TRAIN / LOAD RANDOM FOREST ----------
def get_rf_model(df, table_name, model_path="rf_crr_model.pkl", retrain=False):
    if os.path.exists(model_path) and not retrain:
        print(f"Loading existing model: {model_path}")
        model, feature_names = joblib.load(model_path)
        return model, feature_names

    print("\n🚀 Training new Random Forest Regressor...")
    X = df.drop(columns=["CRR_Score"])
    y = df["CRR_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_test_labels = [risk_category(s) for s in y_test]
    y_pred_labels = [risk_category(s) for s in y_pred]

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=["High Risk", "Medium Risk", "Low Risk"])
    cm_df = pd.DataFrame(cm,
                         index=["Actual High", "Actual Medium", "Actual Low"],
                         columns=["Pred High", "Pred Medium", "Pred Low"])
    print("\n📉 Confusion Matrix:")
    print(cm_df.to_string())
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n🎯 Model Accuracy: {accuracy*100:.2f}%")

    # ---------- Save confusion matrix ----------
    os.makedirs("static/images", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["High Risk", "Medium Risk", "Low Risk"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    plt.title(f"{table_name} Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")
    plt.tight_layout()
    cm_file = f"static/images/{table_name}_confusion_matrix.png"
    plt.savefig(cm_file)
    plt.close()
    print(f"✅ Saved confusion matrix: {cm_file}")

    # Feature importance
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n🌲 Top 10 Most Important Features:")
    print(feat_imp.head(10).to_string(float_format="%.4f"))

    # ---------- Save feature importance ----------
    # Top 10 features
    plt.figure(figsize=(10,6))
    feat_imp.head(10).sort_values().plot(kind='barh', color='skyblue')
    plt.title(f"{table_name} Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    top10_file = f"static/images/{table_name}_rf_feature_importance.png"
    plt.savefig(top10_file)
    plt.close()
    print(f"✅ Saved top 10 RF feature importance: {top10_file}")

    # All features
    plt.figure(figsize=(12,8))
    feat_imp.sort_values().plot(kind='barh', color='lightgreen')
    plt.title(f"{table_name} All Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    full_file = f"static/images/{table_name}_rf_feature_importance_full.png"
    plt.savefig(full_file)
    plt.close()
    print(f"✅ Saved full RF feature importance: {full_file}")

    # Save model
    joblib.dump((model, list(X.columns)), model_path)
    print(f"\n✅ Model saved to {model_path}")
    return model, list(X.columns)

def add_underscores(text):
    return text.replace(" ", "_")


def confusion_by_product_type(df, model, feature_names, table_name):
    """
    Generates confusion matrices + accuracy grouped by ProductType
    for the IndividualCardholders / CompanyCardholders tables.
    Saves each confusion matrix as an image.
    """

    if "ProductType" not in df.columns:
        print(f"❌ No ProductType column in {table_name}. Skipping.")
        return

    os.makedirs("static/images", exist_ok=True)

    products = df["ProductType"].unique()

    print(f"\n===== Confusion Matrices by Product Type for {table_name} =====")

    for product in products:
        df_prod = df[df["ProductType"] == product].copy()

        if len(df_prod) < 10:
            print(f"⚠️ Skipping {product}: not enough samples ({len(df_prod)})")
            continue

        # --- Prepare features like in hybrid_predict ---
        df_features = pd.get_dummies(df_prod, drop_first=False)

        # Add missing dummy columns
        missing_cols = [c for c in feature_names if c not in df_features.columns]
        df_features = pd.concat(
            [df_features, pd.DataFrame(0, index=df_features.index, columns=missing_cols)],
            axis=1
        )
        df_features = df_features[feature_names]

        # --- Predict ---
        y_true = df_prod["RiskCategory"]
        y_pred_raw = model.predict(df_features)
        y_pred = [risk_category(x) for x in y_pred_raw]

        # --- Confusion matrix ---
        labels = ["High Risk", "Medium Risk", "Low Risk"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        accuracy = np.trace(cm) / np.sum(cm)

        print(f"\n📌 {table_name} — Product Type: {product}")
        print(pd.DataFrame(cm, index=labels, columns=labels))
        print(f"✅ Accuracy: {accuracy*100:.2f}%")

        # --- Save figure ---
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

        plt.title(f"{product} Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")
        plt.tight_layout()

        product_adjusted = add_underscores(product)

        file_path = f"static/images/{table_name}_{product_adjusted}_confusion_matrix.png"
        plt.savefig(file_path)
        plt.close()

        print(f"💾 Saved: {file_path}")


# ---------- 5. HYBRID PREDICTION ----------
def hybrid_predict(df, model, feature_names, alpha=0.4):
    df_features = pd.get_dummies(df, drop_first=False)
    missing_cols = [col for col in feature_names if col not in df_features.columns]
    if missing_cols:
        df_features = pd.concat([df_features, pd.DataFrame(0, index=df_features.index, columns=missing_cols)], axis=1)
    df_features = df_features[feature_names]

    preds_rf = model.predict(df_features)
    df['ML_CRR_Pred'] = preds_rf
    df['Hybrid_CRR'] = alpha * df['CRR_Score'] + (1 - alpha) * df['ML_CRR_Pred']
    df['Hybrid_Risk'] = pd.cut(
        df['Hybrid_CRR'],
        bins=[-float('inf'), 40, 65, float('inf')],
        labels=['High Risk', 'Medium Risk', 'Low Risk']
    )
    return df

# ---------- 6. UPDATE DATABASE ----------
def update_table(db_file, table_name, df):
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Updated {table_name} with ML_CRR_Pred, Hybrid_CRR, and Hybrid_Risk.")

# ---------- 7. MAIN ----------
if __name__ == "__main__":
    db_file = "card_risk.db"

    for table in ["IndividualCardholders", "CompanyCardholders"]:
        print(f"\n=== Processing Table: {table} ===")
        df = load_data(db_file, table)
        df_proc = preprocess_data(df)

        # Save missingness heatmap
        #save_missingness_heatmap(df_proc, table)

        # Train/load RF model and save plots
        model, feature_names = get_rf_model(df_proc, table_name=table, retrain=True)
        confusion_by_product_type(df, model, feature_names, table_name=table)

        # Hybrid predictions & update DB
        df_hybrid = hybrid_predict(df, model, feature_names, alpha=0.6)
        update_table(db_file, table, df_hybrid)

