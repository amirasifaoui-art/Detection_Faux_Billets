# Detection_Faux_Billets
Ce projet a pour objectif de construire un modèle prédictif capable de différencier les vrais des faux billets à partir de leurs dimensions géométriques.
# Présentation de l'Application
Une application interactive a été développée avec Streamlit pour permettre aux utilisateurs de charger un fichier de dimensions et d'obtenir une prédiction instantanée.

👉 [Lien vers l'application en ligne] (Ajoutez votre lien Streamlit Cloud ici)

# Compétences Techniques Validées

- Exploration & Visualisation : Analyse descriptive, étude de distribution et tests de normalité des données.
- Algorithmes de Classification : Comparaison de plusieurs modèles : Régression Logistique, K-Nearest Neighbors (KNN) et Random Forest.
- Machine Learning Non Supervisé : Utilisation du K-means pour l'analyse exploratoire et la segmentation.
- Évaluation : Utilisation de matrices de confusion et mesure de la précision pour choisir le modèle le plus performant.

# Structure du Projet

- app.py : Script principal de l'interface Streamlit.
- Analyse_Exploratoire_Donnees_Billets.ipynb : Analyse complète, nettoyage des données et étude des testes statistiques.
- Algorithmes_Billets.ipynb : Comparaisons et entraînement des différents modèles en utilisant plusieurs algorithmes (supervisé et non supervisé) 
- requirements.txt : Liste des bibliothèques nécessaires (Pandas, Scikit-learn, Seaborn, etc.).
- data : Dossier contenant les jeux de données (dimensions des billets).

# Résultats
Le modèle final retenu permet d'identifier les faux billets avec une fiabilité élevée, offrant ainsi un outil d'aide à la décision automatisé.
