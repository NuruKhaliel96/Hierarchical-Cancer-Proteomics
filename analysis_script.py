import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
    from umap import UMAP

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120

# Data loading, merging and filtering

print("Loading and preprocessing data...")

# Load metadata and protein matrix 
try:
    meta_df = pd.read_excel(
        "mmc2.xlsx",
        sheet_name="Cell line level sample info",
        header=1,
        engine="openpyxl"
    )
    data_df = pd.read_excel(
        "mmc3.xlsx",
        sheet_name="Full protein matrix",
        header=1,
        engine="openpyxl"
    )
except Exception:
    meta_df = pd.read_excel("mmc2.xlsx", header=1, engine="openpyxl")
    data_df = pd.read_excel("mmc3.xlsx", header=1, engine="openpyxl")

data_df["model_id_clean"] = data_df["Project_Identifier"].apply(
    lambda x: str(x).split(";")[0]
)

combined_df = pd.merge(
    meta_df[["model_id", "Tissue_type"]],
    data_df,
    left_on="model_id",
    right_on="model_id_clean",
    how="inner"
)

protein_columns = [col for col in combined_df.columns if ";" in str(col)]

print(f"Initial merged dataset: {combined_df.shape[0]} samples, {len(protein_columns)} proteins.")

minimum_sample_threshold = 50
tissue_counts = combined_df["Tissue_type"].value_counts()
valid_tissues = tissue_counts[tissue_counts >= minimum_sample_threshold].index

filtered_df = combined_df[combined_df["Tissue_type"].isin(valid_tissues)].copy()
print(f"After N≥{minimum_sample_threshold} filter: {filtered_df.shape[0]} samples, "
      f"{filtered_df['Tissue_type'].nunique()} tissue types.")

filtered_df[protein_columns] = filtered_df[protein_columns].fillna(0)

# Feature selection
min_samples_quantified = len(filtered_df) * 0.10
protein_sums = filtered_df[protein_columns].astype(bool).sum(axis=0)
selected_proteins = protein_sums[protein_sums > min_samples_quantified].index.tolist()

X = filtered_df[selected_proteins]
y = filtered_df["Tissue_type"]

print(f"Final feature matrix: {X.shape[0]} samples × {X.shape[1]} proteins "
      f"(from {len(protein_columns)} original proteins).")

# Exploratory data analysis (PCA, t-SNE, UMAP)

print("\nRunning PCA, t-SNE and UMAP for EDA...")

# Standardise features for DR methods that are scale-sensitive
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# Combined PCA + t-SNE figure (Figure 1)
plt.figure(figsize=(12, 5))

# Panel A: PCA
plt.subplot(1, 2, 1)
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=y,
    s=30,
    alpha=0.8,
    legend=False
)
plt.title(f"Panel A: PCA (PC1 {pc1_var:.1f}%, PC2 {pc2_var:.1f}%)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Panel B: t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y,
    s=30,
    alpha=0.8,
    legend="full"
)
plt.title("Panel B: t-SNE Embedding")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Tissue type", fontsize="small")

plt.suptitle("Figure 1: Multidimensional Scaling of Proteomic Profiles Across Cancer Cell Lines")
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig("Figure1_PCA_tSNE.png", dpi=300)
plt.show()


# UMAP (Figure 3)
print("Running UMAP...")

umap_model = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(6.5, 5.5))
sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    hue=y,
    s=30,
    alpha=0.8
)
plt.title("Figure 3: UMAP Embedding of Proteomic Profiles by Tissue Type")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Tissue type", fontsize="small")
plt.tight_layout()
plt.savefig("Figure3_UMAP.png", dpi=300)
plt.show()



# Hierarchical random forest classification

print("\nTraining hierarchical Random Forest models...")

# Define lineage label (Haematopoietic vs Solid)
filtered_df["Lineage"] = filtered_df["Tissue_type"].apply(
    lambda t: "Haematopoietic" if t == "Haematopoietic and Lymphoid" else "Solid"
)
y_lineage = filtered_df["Lineage"]

# Align X with filtered_df index just in case
X = filtered_df[selected_proteins]

#Model 1: Lineage Prediction
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X,
    y_lineage,
    test_size=0.3,
    random_state=42,
    stratify=y_lineage
)

rf_lin = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_lin.fit(X_train_lin, y_train_lin)
y_pred_lin = rf_lin.predict(X_test_lin)

lineage_accuracy = accuracy_score(y_test_lin, y_pred_lin)
print(f"Model 1 (Lineage) Test Accuracy: {lineage_accuracy:.4f} "
      f"({lineage_accuracy * 100:.1f}%)")

# Model 2: Subtype Prediction (Solid tissues only) 
solid_samples = filtered_df[filtered_df["Lineage"] == "Solid"].copy()
X_solid = solid_samples[selected_proteins]
y_solid = solid_samples["Tissue_type"]

X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_solid,
    y_solid,
    test_size=0.3,
    random_state=42,
    stratify=y_solid
)

rf_sub = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_sub.fit(X_train_sub, y_train_sub)
y_pred_sub = rf_sub.predict(X_test_sub)

print("\nModel 2 (Subtype, solids only) - Classification Report (Test Split):")
print(classification_report(y_test_sub, y_pred_sub))

# 5-fold Stratified Cross-Validation for Model 2
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_sub, X_solid, y_solid, cv=cv, scoring="accuracy")

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
ci_95 = 1.96 * cv_std  # 95% CI in absolute units

print(f"Model 2 (Subtype) 5-fold CV Accuracy: {cv_mean:.4f} "
      f"({cv_mean * 100:.1f}%)")
print(f"CV Std Dev: {cv_std:.4f}")
print(f"Approx. 95% CI: ±{ci_95:.4f} "
      f"(±{ci_95 * 100:.1f} percentage points)")



# Feature importance plots (Figure 2) 

print("\nComputing feature importances and plotting Figure 2...")

top_n = 10

# Feature importance for Model 1 (Lineage)
importances_lin = rf_lin.feature_importances_
indices_lin = np.argsort(importances_lin)[::-1][:top_n]
top_features_lin = [X.columns[i] for i in indices_lin]
top_importances_lin = importances_lin[indices_lin]

# Clean protein names (remove leading IDs before ';')
top_features_lin_clean = [
    f.split(";")[1] if ";" in f else f for f in top_features_lin
]

# Feature importance for Model 2 (Subtype, solids)
importances_sub = rf_sub.feature_importances_
indices_sub = np.argsort(importances_sub)[::-1][:top_n]
top_features_sub = [X_solid.columns[i] for i in indices_sub]
top_importances_sub = importances_sub[indices_sub]

top_features_sub_clean = [
    f.split(";")[1] if ";" in f else f for f in top_features_sub
]

# Plot bar plots for both (Figure 2)
plt.figure(figsize=(12, 5))

# Panel A: Lineage
plt.subplot(1, 2, 1)
plt.bar(range(top_n), top_importances_lin, align="center")
plt.xticks(range(top_n), top_features_lin_clean, rotation=45, ha="right", fontsize="small")
plt.title("Panel A: Top Lineage-Discriminating Proteins")
plt.ylabel("Random Forest Feature Importance")

# Panel B: Subtype
plt.subplot(1, 2, 2)
plt.bar(range(top_n), top_importances_sub, align="center")
plt.xticks(range(top_n), top_features_sub_clean, rotation=45, ha="right", fontsize="small")
plt.title("Panel B: Top Subtype-Discriminating Proteins (Solid Tissues)")
plt.ylabel("Random Forest Feature Importance")

plt.suptitle("Figure 2: Discriminating Proteins Identified by Hierarchical Random Forest Models")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Figure2_FeatureImportance.png", dpi=300)
plt.show()

print("\nAnalysis complete. Generated:")
print(" - Figure1_PCA_tSNE.png")
print(" - Figure2_FeatureImportance.png")
print(" - Figure3_UMAP.png")
print("and printed Model 1 & Model 2 performance metrics.")
