import streamlit as st
import joblib
import numpy as np
import os

# 1. Charger le cerveau (Modèle) et le traducteur (Scaler)
# Ce bloc essaie tous les noms de fichiers possibles pour éviter les erreurs
@st.cache_resource
def load_models():
    # Liste des noms possibles pour le modèle (avec et sans espace)
    model_names = ['modele_cancer .pkl', 'modele_cancer.pkl']
    scaler_names = ['scaler .pkl', 'scaler.pkl']
    
    model = None
    scaler = None

    # Chargement du modèle
    for name in model_names:
        if os.path.exists(name):
            model = joblib.load(name)
            break
            
    # Chargement du scaler
    for name in scaler_names:
        if os.path.exists(name):
            scaler = joblib.load(name)
            break
            
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Si les fichiers ne sont pas trouvés
if model is None or scaler is None:
    st.error("⚠️ Fichiers manquants ! Vérifiez que 'modele_cancer .pkl' et 'scaler .pkl' sont bien dans l'onglet Files.")
    st.stop()

# 2. Configuration de la page
st.set_page_config(page_title="Détection Cancer IA", page_icon="🏥", layout="wide")

# 3. Titre et description
st.title("🏥 Assistant de Diagnostic Médical (IA)")
st.markdown("""
Cette application utilise l'Intelligence Artificielle pour analyser les mesures d'une tumeur 
et aider les médecins à prédire sa malignité.
""")

# 4. La Barre Latérale (Pour entrer les données)
# AJOUT : Une image de logo
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
st.sidebar.header("🔬 Paramètres de la tumeur")
st.sidebar.write("Ajustez les valeurs ci-dessous :")

def user_input_features():
    # On crée des curseurs pour les 4 valeurs les plus parlantes pour la démo
    radius = st.sidebar.slider('Rayon Moyen (Radius)', 6.0, 30.0, 14.0)
    texture = st.sidebar.slider('Texture Moyenne', 9.0, 40.0, 19.0)
    perimeter = st.sidebar.slider('Périmètre Moyen', 40.0, 190.0, 90.0)
    area = st.sidebar.slider('Surface Moyenne (Area)', 100.0, 2500.0, 600.0)
    
    # Pour que le modèle marche, il lui faut 30 valeurs.
    # Ici, on remplit les 26 autres avec des moyennes standards pour simplifier la démo.
    data_mock = [
        radius, texture, perimeter, area, 
        0.096, 0.104, 0.088, 0.048, 0.181, 0.062, 
        0.40, 1.21, 2.86, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003, 
        16.27, 25.67, 107.2, 880.5, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083
    ]
    
    return np.array(data_mock).reshape(1, -1)

# Récupérer les données
input_data = user_input_features()

# 5. Affichage principal
col1, col2 = st.columns(2)

with col1:
    st.info("Données en entrée (4 principales)")
    st.write(f"- Rayon: {input_data[0][0]}")
    st.write(f"- Texture: {input_data[0][1]}")
    st.write(f"- Périmètre: {input_data[0][2]}")
    st.write(f"- Aire: {input_data[0][3]}")

# 6. Bouton de Prédiction
if st.button('Lancer l\'analyse IA 🔍', use_container_width=True):
    
    # Mise à l'échelle
    input_data_scaled = scaler.transform(input_data)
    
    # Prédiction
    prediction = model.predict(input_data_scaled)
    proba = model.predict_proba(input_data_scaled)[0]
    
    st.divider() # Ligne de séparation
    # AJOUT : Avertissement médical
    st.markdown("---")
    st.warning("⚠️ **AVERTISSEMENT :** Cet outil est une aide au diagnostic basée sur l'IA. Il ne remplace pas l'avis d'un médecin expert.")
    # Affichage du résultat
    if prediction[0] == 1:
        st.error(f"⚠️ RÉSULTAT : Tumeur MALIGNE")
        st.write(f"L'IA est sûre à **{proba[1]*100:.2f}%** qu'il s'agit d'un cancer.")
        st.progress(int(proba[1]*100))
    else:
        st.success(f"✅ RÉSULTAT : Tumeur BÉNIGNE")
        st.write(f"L'IA pense à **{proba[0]*100:.2f}%** que c'est bénin.")
        st.progress(int(proba[0]*100))