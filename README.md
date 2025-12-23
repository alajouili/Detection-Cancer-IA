# Detection-Cancer-IA
Application Streamlit utilisant le Machine Learning pour le diagnostic du cancer du sein
# 🏥 Détection de Cancer du Sein assistée par IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/[VOTRE_NOM_HUGGINGFACE]/[NOM_DU_PROJET])
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)

## 📄 Description
Ce projet est une application de **Machine Learning** destinée à aider le diagnostic médical. Elle permet de prédire si une tumeur mammaire est **bénigne** ou **maligne** en se basant sur des mesures biométriques issues d'une biopsie (FNA).

L'application offre une interface web interactive permettant aux médecins de saisir les caractéristiques de la tumeur et d'obtenir une prédiction instantanée accompagnée d'un indice de confiance.

## 🚀 Démo en ligne
Vous pouvez tester l'application directement ici :
👉 **[CLIQUEZ ICI POUR VOIR L'APPLICATION](https://huggingface.co/spaces/Alajouili123/Detection-Cancer-IA)**
## ⚙️ Fonctionnalités
- **Interface Interactive :** Saisie intuitive des données via des curseurs (Streamlit).
- **Analyse en Temps Réel :** Prédiction instantanée grâce au modèle pré-entraîné.
- **Visualisation :** Affichage clair du résultat (Vert/Rouge) et des probabilités.
- **Données Analysées :** L'IA se base sur des caractéristiques telles que :
  - Le Rayon (Radius)
  - La Texture
  - Le Périmètre
  - La Surface (Area)
  - Et d'autres paramètres morphologiques...

## 🛠️ Technologies Utilisées
* **Python** : Langage principal.
* **Scikit-Learn** : Pour l'entraînement du modèle (Régression Logistique / SVM).
* **Streamlit** : Pour la création de l'interface web (Frontend).
* **Joblib** : Pour la sauvegarde et le chargement des modèles.
* **NumPy** : Pour le calcul scientifique.

## 📂 Structure du Projet
* `app.py` : Le code principal de l'application Streamlit.
* `modele_cancer.pkl` : Le modèle d'IA entraîné.
* `scaler.pkl` : Le scaler pour la normalisation des données.
* `requirements.txt` : Liste des dépendances nécessaires.

## 💻 Installation Locale
Si vous souhaitez lancer ce projet sur votre propre machine :

1. Clonez le dépôt :
```bash
git clone https://github.com/alajouili/Detection-Cancer-IA.git
