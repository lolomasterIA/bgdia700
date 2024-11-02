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

    nb_recipes_by_ingredient = backend.recipe_number_ingredient(session)
    nombre_ingredients = list(nb_recipes_by_ingredient.keys())
    nombre_recettes = list(nb_recipes_by_ingredient.values())

    total_ingredient = len(cook.Ingredient.get_all(session))
    total_recettes = sum(nombre_recettes)

    top_ingredient_used = backend.top_ingredient_used(session, 10)
    df_top_ingredient_used = pd.DataFrame(top_ingredient_used)
    df_top_ingredient_used = df_top_ingredient_used.rename(columns={"name": "Ingrédient",
                                                                    "recipe_count": "Nombre de recettes"})

    ingredient_total_rating, ingredient_review_count = backend.top_ingredient_rating(
        session)

    df_ingredient_review_count = pd.DataFrame(
        list(ingredient_review_count.items()), columns=['Ingrédient', 'nb reviews'])

    # à retravailler car le rating est complexe à appréhender...
    df_ingredient_total_rating = pd.DataFrame(
        list(ingredient_total_rating.items()), columns=['Ingrédient', 'Moyenne rating'])

    # On essait de supprimer les recette qui ont moins de 10 reviews
    df_ingredient_total_rating1 = pd.merge(df_ingredient_total_rating,
                                           df_ingredient_review_count, on="Ingrédient")
    df_ingredient_total_rating1 = df_ingredient_total_rating1[(
        df_ingredient_total_rating1['nb reviews'] > 20)]

    # On essait de supprimer les recette qui ont un rating = 5 exactement
    df_ingredient_total_rating2 = df_ingredient_total_rating1[
        df_ingredient_total_rating1['Moyenne rating'] < 5]

    # on joint les 2 dataframes, nb review et nb recette car ce sont les même
    df_top_ingredient_used = pd.merge(
        df_top_ingredient_used, df_ingredient_review_count.head(10), on='Ingrédient', how='inner')

    menu, col1, col2, col3, col4 = frontend.generate_layout()

    if menu == "Généralité":
        with col1:
            styled_top_10 = df_top_ingredient_used.style.highlight_max(axis=0, color="lightgreen").highlight_min(
                axis=0, color="lightcoral").format({"Nombre": "{:.1f}%"})
            st.subheader(
                "Top 10 des ingrédients (total : " + str(total_ingredient) + ")")
            st.dataframe(styled_top_10, use_container_width=True)

        with col2:
            st.subheader(
                "Nb recettes (total : " + str(total_recettes) + ") / nb ingrédients")
            plt.figure(figsize=(12, 8))
            plt.bar(nombre_ingredients, nombre_recettes)
            plt.xlabel("Nombre d'ingrédients")
            plt.ylabel("Nombre de recettes")
            plt.title("Nombre de recettes en fonction du nombre d'ingrédients")
            plt.xticks(nombre_ingredients)
            plt.xticks(range(1, 41, 2))
            st.pyplot(plt)

        with col3:
            styled_top_10 = df_ingredient_total_rating1.head(10).style.highlight_max(axis=0, color="lightgreen").highlight_min(
                axis=0, color="lightcoral").format({"Nombre": "{:.1f}%"})
            st.subheader(
                "Top 10 des ingrédients avec le meilleur rating (nb > 20)")
            st.dataframe(styled_top_10, use_container_width=True)

        with col4:
            styled_top_10 = df_ingredient_total_rating2.head(10).style.highlight_max(axis=0, color="lightgreen").highlight_min(
                axis=0, color="lightcoral").format({"Nombre": "{:.1f}%"})
            st.subheader(
                "Top 10 des ingrédients avec le meilleur rating (note < 5)")
            st.dataframe(styled_top_10, use_container_width=True)

    elif menu == "Page 2":
        st.subheader(
            "page 2")
        df = backend.generate_kmeans_ingredient(session, 3)
        frontend.display_kmeans_ingredient(df)

    elif menu == "Page 3":
        st.subheader(
            "page 3")
