#  Guide de Déploiement : Streamlit Cloud Community

Ce guide vous explique comment déployer votre modèle de prédiction énergétique gratuitement sur **Streamlit Cloud**.

## 📋 Prérequis

1.  Un compte [GitHub](https://github.com/).
2.  Votre projet doit être poussé (push) sur un dépôt (repository) GitHub.
3.  Le fichier `requirements.txt` doit être présent à la racine de votre projet.

---


## ☁️ Étape 1 : Connexion à Streamlit Cloud

1.  Rendez-vous sur [share.streamlit.io](https://share.streamlit.io/).
2.  Cliquez sur **"Continue with GitHub"**.
3.  Autorisez Streamlit à accéder à vos dépôts.

---

## 🚀 Étape 2 : Déployer l'application

1.  Une fois connecté, cliquez sur le bouton **"New app"**.
2.  **Repository** : Sélectionnez votre dépôt `projet-kaiss`.
3.  **Branch** : Sélectionnez `main` (ou la branche par défaut).
4.  **Main file path** : Saisissez `src/app.py`.
5.  *(Optionnel)* Donnez une URL personnalisée dans "App URL".
6.  Cliquez sur **"Deploy!"**.

---

## ⏳ Étape 3 : Installation et Lancement

Streamlit Cloud va maintenant :
1.  Créer une machine virtuelle.
2.  Installer les dépendances listées dans `requirements.txt`.
3.  Lancer votre application.

Cela peut prendre 2 à 5 minutes lors du premier déploiement.

---
