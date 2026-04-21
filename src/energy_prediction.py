import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ──────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────
df = pd.read_csv('buildingdata.csv')
print("Shape:", df.shape)

# ──────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────
# Supprimer les colonnes inutiles
df = df.drop(columns=['Date', 'Id'])

# Vérification valeurs manquantes (confirmation)
print("\nValeurs manquantes:\n", df.isnull().sum().sum(), "au total")

# Variable cible et features
X = df.drop(columns=['Total electricity consumption'])
y = df['Total electricity consumption']

print("\nFeatures utilisées:", list(X.columns))
print("Variable cible: Total electricity consumption")

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} lignes | Test: {X_test.shape[0]} lignes")

# ──────────────────────────────────────────
# 3. EXPLORATION & VISUALISATION
# ──────────────────────────────────────────

# Distribution de la variable cible
plt.figure(figsize=(8, 4))
plt.hist(y, bins=30, color='steelblue', edgecolor='white')
plt.title("Distribution de la consommation électrique totale")
plt.xlabel("Total electricity consumption")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("distribution_cible.png")
plt.show()

# Heatmap de corrélation
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("heatmap_correlation.png")
plt.show()

# ──────────────────────────────────────────
# 4. MODÉLISATION — 3 modèles comparés
# ──────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

# ──────────────────────────────────────────
# 5. COMPARAISON VISUELLE DES MODÈLES
# ──────────────────────────────────────────
results_df = pd.DataFrame(results).T
print("\n\nRésumé des performances:\n", results_df)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for i, metric in enumerate(["RMSE", "MAE", "R²"]):
    axes[i].bar(results_df.index, results_df[metric], color=['steelblue','darkorange','seagreen'])
    axes[i].set_title(metric)
    axes[i].set_xticklabels(results_df.index, rotation=15)
plt.suptitle("Comparaison des modèles")
plt.tight_layout()
plt.savefig("comparaison_modeles.png")
plt.show()

# ──────────────────────────────────────────
# 6. SAUVEGARDER LE MEILLEUR MODÈLE
# ──────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["R²"])
print(f"\n✅ Meilleur modèle : {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")

best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Modèle sauvegardé dans best_model.pkl")