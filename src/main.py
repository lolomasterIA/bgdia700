"""Le coeur de l'application, s'appuie sur le front et le back."""

from src.logging_config import setup_logging
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.orm import sessionmaker, declarative_base
import src.backend.datalayer.cooking as cook
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import src.backend.backend as backend
import src.frontend.frontend as frontend


# Charger les variables d'environnement
load_dotenv()

# Connexion à la base de données PostgreSQL cooking
DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# Configurer le logger
logger = setup_logging()

# Exemple d'utilisation du logger
# logger.debug("This is a debug message")
# logger.error("This is an error message")
# logger.critical("This is a critical message")

# Votre code principal ici
if __name__ == "__main__":
    logger.info("Application started")
    # ... votre code ...
    recettes = cook.Recipe.get_all(session, n_ingredients=7)
    ingredients = pd.DataFrame()
    for recette in recettes:
        for ingredient in recette.ingredients:
            ingredients.add(ingredient)
    print(len(recettes))
    print(recettes.to_dataframe()["steps"])
"""
    # réduire la taille du jeu de données Recettes / ingrédients
    reducedRecettes = cook.Recipe
    nbingredientsrecettes = {}  # Dictionnaire vide

    totalrecettetest = 0
    for recette in recettes:
        for ingredient in recette.ingredients:
            if ingredient.ingredient_id not in nbingredientsrecettes:
                nbingredientsrecettes[ingredient.ingredient_id] = 1
            else:
                nbingredientsrecettes[ingredient.ingredient_id] += 1

    compteur_valeurs = {}
    compteur_valeursTest = 0
    for valeur in nbingredientsrecettes.values():
        if valeur in compteur_valeurs:
            compteur_valeurs[valeur] += 1
        else:
            compteur_valeurs[valeur] = 1
        compteur_valeursTest += 1
    print(compteur_valeursTest)

    compteur_un = 0
    compteur_sup10000 = 0
    compteur_sup1 = 0
    compteur_sup1000 = 0
    compteur_sup100 = 0
    for ing in compteur_valeurs:
        if compteur_valeurs[ing] == 1:
            compteur_un += 1
        if compteur_valeurs[ing] > 1:
            compteur_sup1 += compteur_valeurs[ing]
        if compteur_valeurs[ing] > 100:
            compteur_sup100 += compteur_valeurs[ing]
        if compteur_valeurs[ing] > 1000:
            compteur_sup1000 += compteur_valeurs[ing]
        if compteur_valeurs[ing] > 10000:
            compteur_sup10000 += compteur_valeurs[ing]
    print("nombre ingrédient dans 1 seule recettes = " + str(compteur_un))
    print("nombre ingrédient dans plus de 10000 recettes = " + str(compteur_sup10000))
    print("nombre ingrédient dans plus de 1000 recettes = " + str(compteur_sup1000))
    print("nombre ingrédient dans plus de 100 recettes = " + str(compteur_sup100))
    print("nombre ingrédient dans plus de 1 recettes = " + str(compteur_sup1))
"""
st.title("Clustering des Ingrédients avec Similarité Cosinus")
k = st.slider("Choisissez le nombre de clusters (k)", 2, 20, 5)

df = backend.generate_kmeans_ingredient(500, recettes, ingredients)
frontend.display_kmeans_ingredient(df, k)
