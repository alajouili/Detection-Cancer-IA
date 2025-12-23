# 🏥 Détection de Cancer du Sein assistée par IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/Alajouili123/Detection-Cancer-IA)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)

## 📄 Description
Ce projet est une application de **Machine Learning** destinée à aider le diagnostic médical. Elle permet de prédire si une tumeur mammaire est **bénigne** ou **maligne** en se basant sur des mesures biométriques issues d'une biopsie (FNA - Fine Needle Aspiration).

L'application offre une interface web interactive permettant aux médecins de saisir les caractéristiques de la tumeur et d'obtenir une prédiction instantanée accompagnée d'un indice de confiance.

## 🚀 Démo en ligne
Vous pouvez tester l'application directement ici :
👉 **[CLIQUEZ ICI POUR VOIR L'APPLICATION](https://huggingface.co/spaces/Alajouili123/Detection-Cancer-IA)**

## 📊 Analyse Exploratoire des Données (EDA)
Avant la modélisation, une analyse approfondie des données a été effectuée :
* **Distribution des classes :** Vérification de l'équilibre entre les cas bénins et malins pour éviter les biais.
* **Corrélations :** Identification des variables (rayon, texture, etc.) ayant le plus fort impact sur le diagnostic via une matrice de corrélation.
* **Nettoyage :** Traitement des valeurs manquantes et vérification de la qualité des données.

## 🧠 Implémentation du Modèle
Le processus de Machine Learning a suivi ces étapes :
1.  **Prétraitement :** Standardisation des données avec `StandardScaler` pour mettre toutes les variables à la même échelle numérique.
2.  **Entraînement :** Utilisation d'un algorithme de classification supervisée (ex: Régression Logistique / SVM) optimisé pour la détection médicale.
3.  **Évaluation :** Le modèle a été validé sur un jeu de test avec une mesure de précision (Accuracy) et l'analyse de la matrice de confusion pour minimiser les faux négatifs.

## ⚙️ Fonctionnalités
- **Interface Interactive :** Saisie intuitive des données via des curseurs (Streamlit).
- **Analyse en Temps Réel :** Prédiction instantanée grâce au modèle pré-entraîné.
- **Visualisation :** Affichage clair du résultat (Vert pour Bénin / Rouge pour Malin) et des probabilités.
- **Paramètres Analysés :** L'IA se base sur des caractéristiques morphologiques telles que le Rayon, la Texture, le Périmètre et la Surface.

## 🚀 Perspectives et Intégration Réelle
**Note sur l'évolution du projet :**

Actuellement, l'application est en mode démonstration avec une saisie de paramètres manuels.

Pour une **intégration hospitalière réelle**, l'architecture cible serait la suivante :
1.  L'application serait connectée directement au logiciel d'analyse d'image du microscope.
2.  Elle récupérerait automatiquement les **30 paramètres géométriques** (Rayon, Texture, Concavité, etc.) via API.
3.  Le diagnostic serait généré instantanément sans intervention humaine ni risque d'erreur de saisie.

## 🛠️ Technologies et Versions
* **Python** (v3.8+) : Langage principal.
* **Scikit-Learn** : Pour l'entraînement du modèle.
* **Streamlit** : Pour l'interface web (Frontend).
* **Pandas & NumPy** : Pour la manipulation des données.
* **Joblib** : Pour la sauvegarde du modèle (`modele_cancer.pkl`) et du scaler (`scaler.pkl`).

## 📂 Structure du Projet
* `app.py` : Le code principal de l'application Streamlit.
* `modele_cancer.pkl` : Le modèle d'IA entraîné.
* `scaler.pkl` : Le scaler pour la normalisation des données.
* `requirements.txt` : Liste des dépendances nécessaires.

## 💻 Installation Locale
Si vous souhaitez lancer ce projet sur votre propre machine :

1. **Clonez le dépôt :**
   ```bash
git clone https://github.com/alajouili/Detection-Cancer-IA.git
