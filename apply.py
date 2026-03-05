import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np


# Config de la page
st.set_page_config(page_title="Projet Détection de Faux Billets", page_icon="ONCFM.png")

# Menu latéral
st.sidebar.title("Menu")
choix = st.sidebar.selectbox("Que voulez-vous faire ?", ["Accueil", "Détection"])

from PIL import Image

# Page Accueil
if choix == "Accueil":
    col1, col2 = st.columns([1, 6])
    with col1:
        img = Image.open("ONCFM.png")
        st.image(img, width=300)
    with col2:
        st.title("Bienvenue chez L’Organisation nationale de lutte contre le faux-monnayage")
        st.write(
            """
    *ONCFM est une organisation publique ayant pour objectif de mettre en place des méthodes d’identification des contrefaçons des billets en euros. Dans le cadre de cette lutte, nous avons développer une application capable de différencier automatiquement les billets authentiques des billets contrefaits.Vous pouvez utiliser cette application via l'onglet "Détection".*
    """
        )
       

# Page Détection
elif choix == "Détection":
    st.title("Détection de Faux Billets")
    st.write("Uploadez un fichier CSV contenant les caractéristiques des billets à analyser.")



    # -------------------------
    # Features utilisées: Bien sélectionner les features utilisées 
    # pour la prédiction 
    # -------------------------
    features_used = [
        'diagonal',
        'height_left',
        'height_right',
        'margin_low',
        'margin_up',
        'length'
         ]
   

    # Upload CSV : Charger un fichier CSV avec les caractéristiques des billets 
    # à analyser 
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        # Charger le CSV
        df = pd.read_csv(uploaded_file)
        st.success("Fichier uploadé avec succès !")
        
        # Afficher les données
        st.subheader("📊 Aperçu des données importées")
        st.dataframe(df.head())


         # Vérification colonnes : Si le fichier ne contient pas les colonnes nécessaires, 
         # afficher un message d'erreur et arrêter l'executuion
        if not all(col in df.columns for col in features_used):
          st.error("❌ Le fichier ne contient pas les colonnes nécessaires.")
          st.write("Colonnes requises :", features_used)
          st.stop()
        
        

        # Charger le modèle: Charger le modèle de régression logistique 
        # préalablement entraîné et sauvegardé avec joblib
        model = joblib.load("modele_regression_with_pipeline.pkl")

        st.title("Prédiction avec Régression Logistique")

         # Sélection des variables : Sélectionner les colonnes du fichier CSV 
         # qui coorespondent aux caractéristiques utilisées pour la prédiction
        X = df[features_used]

        # Bouton de prédiction
        if st.button("Lancer la prédiction"):
           

           # Prédictions: Utiliser le modèle chargé pour faire des prédictions sur
           # les données importées et calculer la probabilité d'être un faux billet
           # pour chaque billet.
           predictions = model.predict(X)
           probabilities = model.predict_proba(X)[:, 1]

           st.success("Prédictions terminées !")

           # Ajout de 2 colonnes: Ajouter les résultats des prédictions 
           # (0 pour faux billet, 1 pour un vrai billet)
           # et la précision dans le Dataframe pour un affichage clair
           df["Prediction"] = predictions
           df["Prediction_Label"] = df["Prediction"].map({0: "Faux", 1: "Vrai"})
           df["Precision (%)"] = (probabilities * 100).round(2)

           # -------------------------
           # Affichage résultats : Afficher les résultats des prédictions dans 
           # un tableau et mettre en évidence les billets détectés comme faux
           # -------------------------
           st.subheader("✅ Résultats des prédictions")
           st.dataframe(df.head())
           faux_billets = df[df["Prediction"] == 0]
           st.subheader(f"🚩 Billets détectés comme faux : {len(faux_billets)}")
           st.dataframe(faux_billets)

           # Afficher le camembert après la prédiction
           if "Prediction_Label" in df.columns:
               fig, ax = plt.subplots(figsize=(7, 4))
               # On force l'ordre pour que 'Faux' soit toujours en premier, puis 'Vrai'
               counts = df["Prediction_Label"].value_counts().reindex(['Faux', 'Vrai'], fill_value=0)
               counts.plot.pie(
                   autopct=lambda x: str(round(x, 2)) + '%',
                   labels=counts.index,
                   ax=ax
               )
               ax.set_ylabel('')
               ax.set_title('Taux des vrais et faux billets')
               st.pyplot(fig)