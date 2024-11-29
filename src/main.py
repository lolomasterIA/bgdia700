"""Le coeur de l'application, s'appuie sur le front et le back."""

from src.logging_config import setup_logging
import streamlit as st
import pandas as pd
from sqlalchemy.orm import sessionmaker, declarative_base
import src.backend.datalayer.cooking as cook
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import src.backend.backend as backend
import src.frontend.frontend as frontend
import plotly.express as px


def load_environment():
    """
    Charge les variables d'environnement à partir d'un fichier .env.

    Retourne:
        dict: Un dictionnaire contenant les variables d'environnement pour la base de données.
    """
    load_dotenv()
    return {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASS": os.getenv("DB_PASS"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_NAME": os.getenv("DB_NAME"),
    }


def create_db_engine(env):
    """
    Crée et retourne un moteur de base de données SQLAlchemy.

    Paramètres:
        env (dict): Un dictionnaire contenant les informations de connexion à la base de données.

    Retourne:
        sqlalchemy.engine.Engine: Un moteur de base de données SQLAlchemy.
    """
    DATABASE_URL = f"postgresql://{env['DB_USER']}:{env['DB_PASS']}@{env['DB_HOST']}:5432/{env['DB_NAME']}"
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session, Base


if __name__ == "__main__":
    env = load_environment()
    engine, session, Base = create_db_engine(env)

    logger = setup_logging()

    logger.info("Application started")

    menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()

    if menu == "Généralité":
        # Affichage des statistiques générales sur les recettes et ingrédients

        # Récupérer le nombre de recettes par nombre d'ingrédients
        nb_recipes_by_ingredient = backend.recipe_number_ingredient(session)
        nombre_ingredients = list(nb_recipes_by_ingredient.keys())
        nombre_recettes = list(nb_recipes_by_ingredient.values())

        # Obtenir le nombre total d'ingrédients et de recettes
        total_ingredient = len(cook.Ingredient.get_all(session))
        total_recettes = sum(nombre_recettes)

        # Top 10 des ingrédients les plus utilisés dans les recettes
        top_ingredient_used = backend.top_ingredient_used(session, 10)
        df_top_ingredient_used = pd.DataFrame(top_ingredient_used)
        df_top_ingredient_used = df_top_ingredient_used.rename(
            columns={"name": "Ingrédient", "recipe_count": "Nombre de recettes"}
        )

        # Notes moyennes et nombre de reviews pour chaque ingrédient
        ingredient_total_rating, ingredient_review_count = (
            backend.top_ingredient_rating(session)
        )

        df_ingredient_review_count = pd.DataFrame(
            list(ingredient_review_count.items()), columns=["Ingrédient", "nb reviews"]
        )
        df_ingredient_total_rating = pd.DataFrame(
            list(ingredient_total_rating.items()),
            columns=["Ingrédient", "Moyenne rating"],
        )
        df_ingredient_total_rating_count = pd.merge(
            df_ingredient_total_rating, df_ingredient_review_count, on="Ingrédient"
        )

        # on joint les 2 dataframes, nb review et nb recette car ce sont les même
        df_top_ingredient_used = pd.merge(
            df_top_ingredient_used,
            df_ingredient_review_count.head(10),
            on="Ingrédient",
            how="inner",
        )

        nb_ing = len(df_ingredient_total_rating_count)

        with col1:
            # Affichage du top 10 des ingrédients les plus utilisés
            styled_top_10 = (
                df_top_ingredient_used.style.highlight_max(axis=0, color="lightgreen")
                .highlight_min(axis=0, color="lightcoral")
                .format({"Nombre": "{:.1f}%"})
            )
            st.subheader(
                "Top 10 des ingrédients (total : " + str(total_ingredient) + ")"
            )
            st.dataframe(styled_top_10, use_container_width=True)

        with col2:
            # Création d'un graphique en barres pour visualiser le nombre de recettes en fonction du nombre d'ingrédients
            st.subheader(
                "Nb recettes (total : " + str(total_recettes) + ") / nb ingrédients"
            )
            fig = px.bar(
                x=nombre_ingredients,
                y=nombre_recettes,
                labels={"x": "Nombre d'ingrédients",
                        "y": "Nombre de recettes"},
            )
            fig.update_xaxes(dtick=2)
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Affichage du top 10 des ingrédients avec le meilleur rating
            st.subheader("Top 10 des ingrédients avec le meilleur rating")
            col41, col42 = st.columns(2)
            with col41:
                min_reviews = st.slider(
                    "Nombre minimum de reviews",
                    min_value=0,
                    max_value=500,
                    value=20,
                    step=1,
                )
            with col42:
                max_rating = st.slider(
                    "Valeur maximale de rating",
                    min_value=0.0,
                    max_value=5.1,
                    value=5.0,
                    step=0.1,
                )
            # On supprime les recettes qui ont moins de min_reviews reviews et moins de max_rating rating
            df_ingredient_total_rating1 = df_ingredient_total_rating_count[
                (df_ingredient_total_rating_count["nb reviews"] > min_reviews)
            ]
            df_ingredient_total_rating2 = df_ingredient_total_rating1[
                df_ingredient_total_rating1["Moyenne rating"] < max_rating
            ]
            styled_top_10 = (
                df_ingredient_total_rating2.head(10)
                .style.highlight_max(axis=0, color="lightgreen")
                .highlight_min(axis=0, color="lightcoral")
                .format({"Nombre": "{:.1f}%"})
            )
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
                    "Moyenne rating": "Évaluation moyenne (rating)",
                },
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "Clusterisation":
        with col1:
            nb_cluster = st.slider(
                "Nombre de clusters", min_value=2, max_value=10, value=2, step=1
            )
            matrix_type = st.selectbox(
                "Type de matrice à utiliser :", options=["tfidf", "count"]
            )
            reduction_type = st.selectbox(
                "Type de réduction de dimensionnalité :", options=["pca", "svd"]
            )
            clustering_type = st.selectbox(
                "Algorithme de clusterisation :",
                options=["kmeans", "dbscan", "agglomerative"],
            )
            st.text("Filtres:")
            st.markdown(
                """
                    * les recettes avec plus de 3 ingrédientss
                    * les recettes avec plus de 15 reviews
                    * les ingrédients qui apparaissent dans moins de 5 recettes
                    * et les ingrédients / recettes associés"""
            )

            df, nombre_total_recettes, nombre_total_ingredients = (
                backend.generate_cluster_recipe(
                    session, matrix_type, reduction_type, clustering_type, 2, nb_cluster
                )
            )

            st.text(
                "Nombre d'ingrédients après filtres : " + str(nombre_total_ingredients)
            )
        with col2:
            frontend.display_cluster_recipe(df)

    elif menu == "Ingrédients qui vont bien ensemble":
        with col1:
            # stokage en session de la grosse matrice pour ne pas la recalculer
            if "co_occurrence_matrix" not in st.session_state:
                co_occurrence_matrix, all_ingredients = backend.generate_matrice_ingredient(
                    session)
                st.session_state.co_occurrence_matrix = co_occurrence_matrix
                st.session_state.all_ingredients = all_ingredients
            else:
                co_occurrence_matrix = st.session_state.co_occurrence_matrix
                all_ingredients = st.session_state.all_ingredients

            st.subheader("Suggestions d'Ingrédients")
            
            # Liste des ingrédients
            selected_ingredient = st.selectbox(
                "Sélectionnez un ingrédient pour obtenir des suggestions :",
                options=all_ingredients,
                placeholder="cheese",
            )
            if selected_ingredient:
                suggestions = backend.suggestingredients(
                    co_occurrence_matrix, selected_ingredient, top_n=5
                )
                if suggestions:
                    st.write(f"Ingrédients qui vont bien avec '{selected_ingredient}':")
                    for ingredient, co_occurrence in suggestions:
                        st.write(
                            f"- {ingredient} : Note {backend.get_ingredient_rating(session, ingredient)} | {co_occurrence} occurrences"
                        )
                else:
                    st.write("Aucune suggestion disponible.")
        with col2:
            if selected_ingredient:
                frontend.display_cloud_ingredient(
                    co_occurrence_matrix, selected_ingredient
                )
    elif menu == "Corrélation minutes":
        with col1:
            selected_model = st.selectbox(
                "Sélectionnez un modèle de prédiction :",
                options=["rl", "xgb", "rf"],
                placeholder="rl",
            )
            selected_method = st.selectbox(
                "Sélectionnez une méthode de nettoyage minutes",
                options=[
                    "DeleteQ1Q3",
                    "Capping",
                    "Log",
                    "Isolation Forest",
                    "DBScan",
                    "Local Outlier Factor",
                ],
                placeholder="deleteQ1Q3",
            )

            mse, r2, coefficients, df_results = backend.generate_regression_minutes(
                session, selected_model, selected_method
            )
            st.write(selected_method)
            st.write("mse = " + str(mse) + " / r2 = " + str(r2))
            st.write("nombre de recettes : " + str(len(df_results)))
            if coefficients is not None:
                st.write(coefficients)
        with col2:
            frontend.display_minutes_byfeature(df_results)

    elif menu == "Corrélation rating ingrédient":
        with col1:
            selected_model = st.selectbox(
                "Sélectionnez un modèle de prédiction :",
                options=["rl", "xgb", "rf"],
                placeholder="rl",
            )

            mse, r2, coefficients, df_results = backend.generate_regression_ingredient(
                session, selected_model
            )
            st.write("mse = " + str(mse) + " / r2 = " + str(r2))
            if coefficients is not None:
                st.write(coefficients)
        with col2:
            frontend.display_rating_ingredientbyfeature(df_results)
