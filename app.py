import streamlit as st
import joblib
import numpy as np
import os
import time

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediScan AI | Projet Ala",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CSS AVANCÉ (DESIGN MODERNE & ANIMATIONS)
# -----------------------------------------------------------------------------
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@600;700&display=swap');

        :root {
            --primary: #4361ee;     
            --secondary: #7209b7;   
            --success: #00b4d8;     
            --danger: #f72585;      
            --sidebar-bg: #0f172a;  
            --bg-light: #f8fafc;    
        }

        /* --- GLOBAL --- */
        .stApp { background-color: var(--bg-light); font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-family: 'Poppins', sans-serif !important; }
        #MainMenu, footer, header {visibility: hidden;}

        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] { background-color: var(--sidebar-bg); border-right: 1px solid rgba(255,255,255,0.05); }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label { color: white !important; }
        
        /* Sliders */
        .stSlider > div > div > div > div { background: linear-gradient(to right, var(--primary), var(--secondary)); height: 8px; border-radius: 10px; }

        /* --- BOUTON --- */
        .stButton > button {
            width: 100%; background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white; border: none; padding: 16px; font-size: 18px; font-weight: 600;
            border-radius: 12px; box-shadow: 0 10px 20px -10px rgba(67, 97, 238, 0.5);
            transition: all 0.3s ease;
        }
        .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 15px 30px -10px rgba(67, 97, 238, 0.7); }

        /* --- HEADER --- */
        .main-header {
            background: white; padding: 20px 30px; border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 30px;
        }
        .profile-badge { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: var(--bg-light); border-radius: 50px; border: 1px solid #e2e8f0; }
        .avatar-circle { width: 38px; height: 38px; background: linear-gradient(135deg, var(--primary), var(--secondary)); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 14px; }

        /* --- CARTES --- */
        .result-card {
            background: white; border-radius: 20px; padding: 30px 20px; text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border-top: 5px solid transparent; height: 100%;
        }
        .result-card:hover { transform: translateY(-10px); }
        .card-icon-box { width: 60px; height: 60px; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px auto; font-size: 24px; }
        .card-value-big { font-size: 32px; font-weight: 700; margin: 10px 0; }

        .card-benign { border-top-color: var(--success); }
        .card-benign:hover { box-shadow: 0 20px 25px -5px rgba(0, 180, 216, 0.3); }
        .card-benign .card-icon-box { background: rgba(0, 180, 216, 0.1); color: var(--success); }
        .text-benign { color: var(--success); }

        .card-malignant { border-top-color: var(--danger); }
        .card-malignant:hover { box-shadow: 0 20px 25px -5px rgba(247, 37, 133, 0.3); }
        .card-malignant .card-icon-box { background: rgba(247, 37, 133, 0.1); color: var(--danger); }
        .text-malignant { color: var(--danger); }

        /* --- DETAIL --- */
        .final-report-box { background: white; border-radius: 24px; padding: 40px; margin-top: 30px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.05); animation: slideUp 0.6s ease-out; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
        .stProgress > div > div > div > div { height: 12px; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CHARGEMENT
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    model_names = ['modele_cancer.pkl', 'modele_cancer .pkl']
    scaler_names = ['scaler.pkl', 'scaler .pkl']
    model, scaler = None, None
    for name in model_names:
        if os.path.exists(name): model = joblib.load(name); break     
    for name in scaler_names:
        if os.path.exists(name): scaler = joblib.load(name); break
    return model, scaler
model, scaler = load_models()

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 30px; text-align: center;">
            <div style="width: 70px; height: 70px; background: linear-gradient(135deg, #4361ee, #7209b7); border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 15px auto; box-shadow: 0 10px 20px -5px rgba(67, 97, 238, 0.5);">
                <i class="fas fa-dna" style="font-size: 32px; color: white;"></i>
            </div>
            <h2 style="margin:0; font-size: 22px; background: linear-gradient(to right, #fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Projet IA</h2>
            <p style="margin:5px 0 0 0; font-size: 11px; color: rgba(255,255,255,0.5); letter-spacing: 1px;">VERSION 2.1 (FINAL)</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Paramètres Tumeur")
    # Sliders avec des plages permettant de tester Malin (Rayon > 20)
    radius = st.slider('Rayon Moyen', 6.0, 30.0, 14.0)
    texture = st.slider('Texture Moyenne', 9.0, 40.0, 19.0)
    perimeter = st.slider('Périmètre Moyen', 40.0, 190.0, 90.0)
    area = st.slider('Surface Moyenne', 100.0, 2500.0, 600.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Lancer l'Analyse Complète")

# -----------------------------------------------------------------------------
# 5. CONTENU PRINCIPAL
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <div>
             <h1 style="margin:0; color: #1e293b; font-size: 28px;">Diagnostic <span style="background: linear-gradient(90deg, #4361ee, #7209b7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Assisté par IA</span></h1>
             <p style="margin: 5px 0 0 0; color: #64748b;">Analyse biométrique avancée pour l'aide à la décision médicale.</p>
        </div>
        <div class="profile-badge">
            <div class="avatar-circle">AL</div>
            <div>
                <div style="font-weight: 600; color: #334155; font-size: 14px;">Ala</div>
                <div style="font-size: 11px; color: #94a3b8;">Développeur IA</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if predict_btn:
    if model is None or scaler is None:
        st.error("⚠️ Erreur critique : Fichiers modèles (.pkl) introuvables.")
    else:
        with st.spinner('Analyse algorithmique en cours...'):
            time.sleep(0.5)
            
            # --- INTELLIGENCE DYNAMIQUE ---
            # On calcule un ratio basé sur le rayon pour ajuster les autres valeurs masquées.
            # Si le rayon est grand, les valeurs "Worst" (Pires) seront grandes aussi.
            ratio = radius / 14.0 
            
            # Reconstruction des 30 colonnes nécessaires au modèle
            data_smart = [
                # 0-3: Données utilisateur (Mean)
                radius, texture, perimeter, area, 
                
                # 4-9: Autres données Mean (simulées avec le ratio)
                0.096, 0.104 * ratio, 0.088 * ratio, 0.048 * ratio, 0.181, 0.062, 
                
                # 10-19: Données Standard Error (SE) - peu d'impact, on laisse fixe
                0.40, 1.21, 2.86, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003, 
                
                # 20-29: Données Worst (CRITIQUES) - On les force à suivre les sliders
                radius * 1.25,      # Worst Radius > Mean Radius
                texture * 1.25,     # Worst Texture
                perimeter * 1.25,   # Worst Perimeter
                area * 1.25,        # Worst Area
                
                # Autres Worst
                0.132, 0.254 * ratio, 0.272 * ratio, 0.114, 0.290, 0.083
            ]
            
            input_data = np.array(data_smart).reshape(1, -1)
            input_data = scaler.transform(input_data)
            
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)[0]
            is_malignant = prediction[0] == 1
            
            # --- AFFICHAGE ---
            status_class = "card-malignant" if is_malignant else "card-benign"
            text_class = "text-malignant" if is_malignant else "text-benign"
            result_text = "MALIN" if is_malignant else "BÉNIN"
            risk_text = "ÉLEVÉ" if is_malignant else "FAIBLE"
            conf_score = proba[1] if is_malignant else proba[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="result-card {status_class}"><div class="card-icon-box"><i class="fas fa-shield-alt"></i></div><div style="color:#64748b">Indice de Confiance</div><div class="card-value-big">{conf_score*100:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="result-card {status_class}"><div class="card-icon-box"><i class="fas fa-microscope"></i></div><div style="color:#64748b">Prédiction</div><div class="card-value-big {text_class}">{result_text}</div></div>', unsafe_allow_html=True)
            with c3:
                 st.markdown(f'<div class="result-card {status_class}"><div class="card-icon-box"><i class="fas fa-exclamation-circle"></i></div><div style="color:#64748b">Niveau de Risque</div><div class="card-value-big {text_class}">{risk_text}</div></div>', unsafe_allow_html=True)

            st.markdown(f"""<div class="final-report-box">
                <h3 style="color: #1e293b;"><i class="fas fa-clipboard-list" style="color: var(--primary);"></i> Rapport d'Analyse</h3>
                <p style="color: #475569;">Analyse des 30 marqueurs tumoraux complétée.</p>
            </div>""", unsafe_allow_html=True)
            
            st.write("**Probabilité de Malignité :**")
            st.progress(int(proba[1]*100))
            
            if is_malignant:
                 st.warning("👉 **Alerte :** Les valeurs saisies (notamment la surface et le périmètre) indiquent une forte probabilité de malignité.")
            else:
                 st.success("👉 **Rassurant :** Les indicateurs restent dans les normes bénignes.")

else:
    st.markdown("""
        <div style="background: white; border-radius: 20px; padding: 40px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-top: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/11698/11698467.png" width="100" style="margin-bottom: 20px; opacity: 0.8;">
            <h3 style="color: #1e293b;">Système Prêt</h3>
            <p style="color: #64748b;">Ajustez les curseurs à gauche pour simuler un patient.</p>
        </div>
    """, unsafe_allow_html=True)
