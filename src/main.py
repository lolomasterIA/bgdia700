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
import plotly.express as px


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

    menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()

    if menu == "Généralité":
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
        df_ingredient_total_rating_count = pd.merge(df_ingredient_total_rating,
                                                    df_ingredient_review_count, on="Ingrédient")

        # on joint les 2 dataframes, nb review et nb recette car ce sont les même
        df_top_ingredient_used = pd.merge(
            df_top_ingredient_used, df_ingredient_review_count.head(10), on='Ingrédient', how='inner')

        nb_ing = len(df_ingredient_total_rating_count)

        with col1:
            styled_top_10 = df_top_ingredient_used.style.highlight_max(axis=0, color="lightgreen").highlight_min(
                axis=0, color="lightcoral").format({"Nombre": "{:.1f}%"})
            st.subheader(
                "Top 10 des ingrédients (total : " + str(total_ingredient) + ")")
            st.dataframe(styled_top_10, use_container_width=True)

        with col2:
            # Créer le graphique en barres avec Plotly
            st.subheader(
                "Nb recettes (total : " + str(total_recettes) + ") / nb ingrédients")
            fig = px.bar(
                x=nombre_ingredients,
                y=nombre_recettes,
                labels={"x": "Nombre d'ingrédients",
                        "y": "Nombre de recettes"},
                title="Nombre de recettes en fonction du nombre d'ingrédients"
            )

            # Ajuster l'axe X pour afficher tous les deux chiffres
            fig.update_xaxes(dtick=2)

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Top 10 des ingrédients avec le meilleur rating")
            col41, col42 = st.columns(2)
            with col41:
                min_reviews = st.slider(
                    "Nombre minimum de reviews", min_value=0, max_value=500, value=20, step=1)
            with col42:
                max_rating = st.slider(
                    "Valeur maximale de rating", min_value=0.0, max_value=5.1, value=5.0, step=0.1)
            # On supprime les recettes qui ont moins de min_reviews reviews et moins de max_rating rating
            df_ingredient_total_rating1 = df_ingredient_total_rating_count[(
                df_ingredient_total_rating_count['nb reviews'] > min_reviews)]
            df_ingredient_total_rating2 = df_ingredient_total_rating1[
                df_ingredient_total_rating1['Moyenne rating'] < max_rating]
            styled_top_10 = df_ingredient_total_rating2.head(10).style.highlight_max(axis=0, color="lightgreen").highlight_min(
                axis=0, color="lightcoral").format({"Nombre": "{:.1f}%"})
            st.dataframe(styled_top_10, use_container_width=True)
        with col4:
            # ingrédients selon moyenne de rating et nombre de rating
            st.subheader("ingrédients selon le rating et le nombre de reviews")
            nb_ing = len(df_ingredient_total_rating2)
            st.text("Nombre d'ingrédient : " + str(nb_ing))
            fig = px.scatter(
                df_ingredient_total_rating2,
                x="nb reviews",
                y="Moyenne rating",
                labels={
                    "nb reviews": "Nombre de reviews",
                    "Moyenne rating": "Évaluation moyenne (rating)"
                },
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "Clusterisation":
        with col1:
            st.subheader("Kmeans sur les ingredients dans les recettes")
            st.text("Filtres : ")
            st.text("Nombre de recettes après filtres : ")
            st.text("Nombre d'ingrédients après filtres : ")
            df = backend.generate_kmeans_ingredient(session, 3)
            frontend.display_kmeans_ingredient(df)

    elif menu == "Page 3":
        st.subheader(
            "page 3")
