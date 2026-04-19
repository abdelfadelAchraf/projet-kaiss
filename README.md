# 🏢 Prédiction de la Consommation Énergétique des Bâtiments
### Projet Data Mining — Python & Machine Learning

---

## 👥 Équipe & Contexte

| Champ | Détail |
|---|---|
| **Sujet** | Prédiction de la consommation énergétique des bâtiments |
| **Type de problème** | Régression supervisée |
| **Dataset** | [Building Energy Consumption — Kaggle](https://www.kaggle.com/code/varat7v2/building-energy-consumption) |
| **Outils** | Python, VSCode, Git/GitHub |
| **Librairies** | pandas, numpy, matplotlib, seaborn, scikit-learn, joblib |

---

## 📁 Structure du Projet

```
projet-kaiss/
│
├── buildingdata.csv          # Dataset brut
├── energy_prediction.py      # Script principal (preprocessing + modélisation)
├── best_model.pkl            # Modèle sauvegardé (Random Forest ou autre)
├── scaler.pkl                # Scaler sauvegardé pour la normalisation
│
├── distribution_cible.png    # Graphique : distribution de la variable cible
├── heatmap_correlation.png   # Graphique : matrice de corrélation
├── comparaison_modeles.png   # Graphique : comparaison des performances
│
└── README.md                 # Ce fichier de documentation
```

---

## 📊 Étape 1 — Compréhension du Problème

### Objectif
Prédire la **consommation électrique totale** (`Total electricity consumption`) d'un bâtiment en fonction de ses caractéristiques physiques et environnementales.

Il s'agit d'un problème de **régression** : la variable cible est une valeur numérique continue.

### Pourquoi ce problème est important ?
La consommation énergétique des bâtiments représente une part significative de la consommation mondiale d'énergie. Pouvoir la prédire avec précision permet :
- d'optimiser la gestion de l'énergie,
- de réduire les coûts et l'empreinte carbone,
- de guider les décisions architecturales et d'isolation.

---

## 🗃️ Étape 2 — Collecte des Données

Le dataset provient de **Kaggle** et a été téléchargé au format CSV (`buildingdata.csv`).

### Aperçu général

| Propriété | Valeur |
|---|---|
| Nombre de lignes | 1023 |
| Nombre de colonnes | 26 |
| Valeurs manquantes | **0** (dataset propre) |
| Types de données | 23 float64, 2 int64, 1 object (Date) |

### Description des colonnes

| # | Colonne | Type | Rôle |
|---|---|---|---|
| 0 | `Date` | object | Identifiant temporel — **supprimé** |
| 1 | `Id` | int64 | Identifiant ligne — **supprimé** |
| 2 | `Total electricity consumption` | float64 | ⭐ **Variable cible** |
| 3 | `Air Temperature` | float64 | Feature |
| 4 | `Radiant Temperature` | float64 | Feature |
| 5 | `Operative Temperature` | float64 | Feature |
| 6 | `Outside Dry-Bulb Temperature` | float64 | Feature |
| 7 | `Glazing` | float64 | Feature (isolation vitrée) |
| 8 | `Walls` | float64 | Feature (isolation murs) |
| 9 | `Ceilings (int)` | float64 | Feature (isolation plafonds) |
| 10 | `Floors (int)` | float64 | Feature (isolation sols) |
| 11 | `Ground Floors` | float64 | Feature |
| 12 | `Partitions (int)` | float64 | Feature |
| 13 | `Roofs` | float64 | Feature (isolation toiture) |
| 14 | `External Infiltration` | float64 | Feature |
| 15 | `External Vent.` | float64 | Feature (ventilation) |
| 16 | `General Lighting` | float64 | Feature (éclairage) |
| 17 | `Computer + Equip` | float64 | Feature (équipements) |
| 18 | `Occupancy` | float64 | Feature (occupation) |
| 19 | `Solar Gains Interior Windows` | float64 | Feature |
| 20 | `Solar Gains Exterior Windows` | float64 | Feature |
| 21 | `Zone Sensible Heating` | int64 | Feature |
| 22 | `Zone Sensible Cooling` | float64 | Feature |
| 23 | `Sensible Cooling` | float64 | Feature |
| 24 | `Total Cooling` | float64 | Feature |
| 25 | `Mech Vent + Nat Vent + Infiltration` | float64 | Feature |

---

## 🧹 Étape 3 — Prétraitement des Données

### Actions effectuées

**Suppression des colonnes non pertinentes**
```python
df = df.drop(columns=['Date', 'Id'])
```
- `Date` : colonne texte sans valeur prédictive directe dans ce contexte
- `Id` : simple identifiant de ligne, ne représente aucune information physique

**Vérification des valeurs manquantes**
```python
df.isnull().sum().sum()  # Résultat : 0
```
Aucune valeur manquante détectée — aucun remplissage nécessaire.

**Séparation features / variable cible**
```python
X = df.drop(columns=['Total electricity consumption'])
y = df['Total electricity consumption']
```

**Normalisation (StandardScaler)**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
La normalisation centre chaque feature à une moyenne de 0 et un écart-type de 1. Elle est indispensable pour des algorithmes sensibles aux échelles (notamment la régression linéaire et le KNN).

**Découpage Train / Test**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```
| Ensemble | Taille |
|---|---|
| Entraînement | 818 lignes (80%) |
| Test | 205 lignes (20%) |

---

## 📈 Étape 4 — Exploration & Visualisation

### Distribution de la variable cible
Le graphique `distribution_cible.png` montre la répartition des valeurs de consommation électrique. Cela permet de détecter d'éventuels outliers et de comprendre si la distribution est normale ou asymétrique.

### Matrice de corrélation
Le graphique `heatmap_correlation.png` affiche les corrélations entre toutes les variables. Les features fortement corrélées à `Total electricity consumption` sont les plus utiles pour la prédiction.

**Interprétation attendue :**
- Les features de température (Air, Operative, Dry-Bulb) sont généralement corrélées à la consommation
- Les gains solaires et l'occupation influencent directement la demande énergétique
- Des corrélations très élevées entre features (multicolinéarité) peuvent indiquer des features redondantes

---

## 🤖 Étape 5 — Modélisation

Trois algorithmes de régression ont été entraînés et comparés :

### 1. Régression Linéaire (`LinearRegression`)
- **Principe** : modélise une relation linéaire entre les features et la cible
- **Avantage** : simple, rapide, interprétable
- **Limite** : ne capte pas les relations non-linéaires

### 2. Arbre de Décision (`DecisionTreeRegressor`)
- **Principe** : divise récursivement les données selon des seuils sur les features
- **Avantage** : interprétable, capte la non-linéarité
- **Limite** : tendance à l'overfitting sur les données d'entraînement

### 3. Random Forest (`RandomForestRegressor`)
- **Principe** : ensemble de 100 arbres de décision, agrège leurs prédictions
- **Avantage** : robuste, meilleure généralisation, moins d'overfitting
- **Limite** : moins interprétable, plus lent à entraîner

```python
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}
```

---

## 📐 Étape 6 — Évaluation des Modèles

### Métriques utilisées

| Métrique | Formule | Interprétation |
|---|---|---|
| **RMSE** | √(MSE) | Erreur moyenne en unités de la cible — plus bas = mieux |
| **MAE** | moyenne(|y - ŷ|) | Erreur absolue moyenne — robuste aux outliers |
| **R²** | 1 - SS_res/SS_tot | Part de variance expliquée — plus proche de 1 = mieux |

### Résultats obtenus

> *(Les valeurs seront remplies après exécution du script)*

| Modèle | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Decision Tree | — | — | — |
| Random Forest | — | — | — |

Le graphique `comparaison_modeles.png` visualise ces métriques côte à côte.

### Critère de sélection du meilleur modèle
Le modèle avec le **R² le plus élevé** est automatiquement sélectionné et sauvegardé :
```python
best_model_name = max(results, key=lambda k: results[k]["R²"])
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

## 💾 Sauvegarde du Modèle

Le meilleur modèle et le scaler sont sérialisés avec `joblib` pour une réutilisation future (notamment lors du déploiement) :

- `best_model.pkl` → le modèle entraîné
- `scaler.pkl` → le scaler pour transformer les nouvelles entrées de la même façon

---

## 🚀 Étape 7 (Bonus) — Déploiement avec Streamlit

> *(En cours — à venir)*

Un fichier `app.py` sera créé pour permettre à n'importe quel utilisateur de saisir les caractéristiques d'un bâtiment et d'obtenir une prédiction de sa consommation électrique via une interface web simple.

**Technologies prévues :**
- [Streamlit](https://streamlit.io) pour l'interface utilisateur
- [Streamlit Community Cloud](https://streamlit.io/cloud) pour l'hébergement gratuit (connexion au repo GitHub)

---

## ▶️ Comment exécuter le projet

### Prérequis
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Lancer le script principal
```bash
python energy_prediction.py
```

### Résultats générés
- Affichage des métriques dans le terminal
- 3 graphiques sauvegardés automatiquement (`.png`)
- Modèle sauvegardé (`best_model.pkl`)

---

## 🔄 Suivi de l'avancement

| Étape | Statut |
|---|---|
| 1. Compréhension du problème | ✅ Terminé |
| 2. Collecte des données | ✅ Terminé |
| 3. Prétraitement | ✅ Terminé |
| 4. Exploration & Visualisation | ✅ Terminé |
| 5. Modélisation | ✅ Terminé |
| 6. Évaluation | ✅ Terminé |
| 7. Interprétation & Conclusion | 🔄 En cours |
| 8. Déploiement Streamlit (bonus) | ⏳ À faire |

---

## 📝 Conclusion

*(À compléter après analyse des résultats)*

Ce projet applique une démarche complète de Data Mining sur un dataset de consommation énergétique de bâtiments. En comparant plusieurs algorithmes de régression, nous identifions le modèle le plus performant et le rendons réutilisable via une sauvegarde `.pkl`. L'objectif final est de déployer ce modèle dans une application web accessible.