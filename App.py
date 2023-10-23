# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import pandas as pd
import io
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

def main_page():
    #st.markdown("# Main page 🎈")
    st.sidebar.markdown("# ACCUEIL 🎈")
    st.header ("Auteur : Cissé NIANG")
    st.title("Application de Machine Learning sur les dépenses médicales en fonction de l'âge, l'IMC, le sexe.... ")
    choix = st.sidebar.radio("Sélection", ["Description", "Documentation"])
    if choix == "Description":
        st.subheader("Voici la description des données")
        st.write("L'ensemble de données insurance.csv contient 1338 observations (lignes) et 7 caractéristiques (colonnes). L'ensemble de données contient 4 caractéristiques numériques (âge, bmi, enfants et dépenses) et 3 caractéristiques nominales (sexe, fumeur et région) qui ont été converties en facteurs avec une valeur numérique désignée pour chaque niveau.")
        st.write("L'objectif de cet exercice est d'examiner différentes caractéristiques pour observer leur relation et de tracer une régression linéaire multiple basée sur plusieurs caractéristiques de l'individu telles que l'âge, l'état physique/familial et l'emplacement par rapport à leurs frais médicaux existants à utiliser pour prédire les frais médicaux futurs des personnes qui aident l'assurance médicale à prendre une décision sur la perception de la prime")
    if choix == "Documentation":
        st.subheader("Voici la documentation de l'application Web")
        st.write("1) https://blog.streamlit.io/introducing-multipage-apps/")
        st.write("2) https://chat.openai.com/")
        st.write("3) https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction")

chemin = "/Users/cisseniang/Documents/Data/Données ML/" 
path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'

# Concaténez le chemin pour obtenir le chemin absolu du fichier 'photo.png'
chemin = os.path.join(path, 'Insurance.csv')

def page2():
    st.markdown("# Analyse Exploratoire des données❄️")
    st.sidebar.markdown("# EDA ❄️")
    choix = st.sidebar.radio("Sélection", ["Data et Infos", "Analyse descriptive"])
 

    if choix == "Data et Infos":
        st.subheader("Afficher les données")
    #def load_data(): 
        #chemin = "/Users/cisseniang/Documents/Data/Données ML/Insurance.csv"
    
        # Obtenez le chemin du répertoire courant
     
        with open(chemin, 'rb') as fichier:
            # Charger le modèle
            data = pd.read_csv(fichier)

    #Affichage de la table de données
        #df = load_data()
 
        df_sample = data.sample(100)
        #st.write(df)
        st.dataframe(df_sample)
        st.button("Rerun")

        
        st.subheader("Infos sur la data")

        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    if choix == "Analyse descriptive":
        st.subheader("Statistiques descriptives")
        

        with open(chemin, 'rb') as fichier:
            # Charger le modèle
            data = pd.read_csv(fichier)
        st.dataframe(data.describe())


    
def page3():
    st.markdown("# Modélisation 🎉")
    st.sidebar.markdown("# Modélisation🎉")

    choix = st.sidebar.radio("choix du modèle", ["Modèle de Régression", "Modèle Ridge"])

    if choix == "Modèle Ridge":

 
        # Charger le modèle pré-entraîné
        #modele_chemin1 = "/Users/cisseniang/Documents/Data/Données ML/model.pkl"
        #model_ridge = os.path.join(path, 'model.pkl')
        model_ridge = joblib.load(open(path+'/model.pkl', 'rb'))


        # Interface utilisateur Streamlit
        st.title("Prédiction de Dépenses Médicales")

        st.write(
            "Cette application permet de prédire les dépenses médicales en fonction de l'âge, du sexe, "
            "de l'IMC, du nombre d'enfants, du statut de fumeur et de la région."
        )

        # Entrées utilisateur
        age = st.slider("Âge", min_value=0, max_value=100, value=30, step=1)
        sex = st.selectbox("Sexe", ['male', 'female'])
        bmi = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        children = st.slider("Nombre d'enfants", min_value=0, max_value=5, value=0, step=1)
        smoker = st.radio("Fumeur", ['yes', 'no'])
        region = st.selectbox("Région", ['southeast', 'southwest', 'northwest', 'northeast'])

        # Prédictions
        input_features = [[age, bmi, children, 1 if sex == 'male' else 0, 1 if smoker == 'yes' else 0,
                        1 if region == 'southeast' else 0]]
        prediction = model_ridge.predict(input_features)

        # Afficher le résultat
        st.write(f"La prédiction des dépenses médicales est : {prediction[0]:.2f} $")

    if choix == "Modèle de Régression":

 
        model_reg = joblib.load(open(path+'/model_reg.pkl', 'rb'))

        # Interface utilisateur Streamlit
        st.title("Prédiction de Dépenses Médicales")

        st.write(
            "Cette application permet de prédire les dépenses médicales en fonction de l'âge, du sexe, "
            "de l'IMC, du nombre d'enfants, du statut de fumeur et de la région."
        )

        # Entrées utilisateur
        age = st.slider("Âge", min_value=0, max_value=100, value=30, step=1)
        sex = st.selectbox("Sexe", ['male', 'female'])
        bmi = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        children = st.slider("Nombre d'enfants", min_value=0, max_value=5, value=0, step=1)
        smoker = st.radio("Fumeur", ['yes', 'no'])
        region = st.selectbox("Région", ['southeast', 'southwest', 'northwest', 'northeast'])

        # Prédictions
        input_features = [[age, bmi, children, 1 if sex == 'male' else 0, 1 if smoker == 'yes' else 0,
                        1 if region == 'southeast' else 0]]
        prediction = model_reg.predict(input_features)

        # Afficher le résultat
        st.write(f"La prédiction des dépenses médicales est : {prediction[0]:.2f} $")



page_names_to_funcs = {
    "Accueil": main_page,
    "EDA": page2,
    "Modélisation": page3,
}

selected_page = st.sidebar.selectbox("Sélectionner une page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()