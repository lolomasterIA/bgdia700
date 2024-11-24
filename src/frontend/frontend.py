"""
Frontend va fournir l'ensemble des méthodes permettant l'affichage.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_layout():
    """
    Configure et génère la mise en page principale de l'application Streamlit.

    Retourne:
    - tuple : Contient le menu sélectionné et les objets de colonne pour structurer le contenu dans des sections.
    """
    # Configurer la page pour un affichage en pleine largeur
    st.set_page_config(
        page_title="Mange ta main",
        page_icon="src/frontend/images/favicon.png",  # Chemin relatif vers l'icône
        layout="wide",
    )
    # Titre principal de l'application
    st.image("src/frontend/images/mangetamain.jpg")

    menu = st.selectbox(
        "", ["Généralité", "Clusterisation", "Ingrédients qui vont bien ensemble", "Corrélation rating ingrédient", "Corrélation minutes"])
    )

    # Zone principale de contenu
    st.header(menu)

    # Utilisation des colonnes pour diviser la zone centrale en plusieurs sections
    with st.container():
        if menu == "Clusterisation" or menu == "Ingrédients qui vont bien ensemble" or menu == "Corrélation rating ingrédient" or menu == "Corrélation minutes":
            col1, col2 = st.columns([1, 2])
        else:
            col1, col2 = st.columns(2)

    with st.container():
        col3, col4 = st.columns(2)

    with st.container():
        col5, col6 = st.columns(2)

    # Footer ou informations supplémentaires
    st.markdown("---")
    st.text("powered by Telecom Paris Master IA")

    return menu, col1, col2, col3, col4, col5, col6


### Travaux sur les ingrédients ###
def display_cluster_recipe(df_recipes_ingredients):
    """
    Affiche les clusters d'ingrédients en fonction des recettes sous forme de graphique interactif.

    Paramètres:
    - df_recipes_ingredients : DataFrame contenant les informations de clusterisation et les coordonnées PCA.

    Cette fonction génère un graphique de dispersion où chaque point représente un ingrédient, coloré selon le cluster.
    """
    fig = px.scatter(
        df_recipes_ingredients,
        x="pca_x",
        y="pca_y",
        color="cluster",
        hover_data=["recette"],
        title="Cluster des ingrédients en fonction des recettes",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig)


def display_kmeans_ingredient(df_recipes_ingredients):
    """
    Affiche les clusters d'ingrédients en fonction des recettes sous forme de graphique interactif.

    Paramètres:
    - df_recipes_ingredients : DataFrame contenant les informations de clusterisation et les coordonnées PCA.

    Cette fonction génère un graphique de dispersion où chaque point représente un ingrédient, coloré selon le cluster.
    """
    fig = px.scatter(
        df_recipes_ingredients,
        x="x",
        y="y",
        color="cluster",
        hover_data=["ingredient"],
        title="Cluster des ingrédients en fonction des recettes",
    )
    st.plotly_chart(fig)


def display_cloud_ingredient(co_occurrence_matrix, selected_ingredient):
    """
    Tentative d'afficher un ingredient avec la distance des autres ingrédients.

    Paramètres:
    - co_occurrence_matrix: np.array des co occurrences des ingrédients
    - selected_ingredient: l'ingredient à comparer
    """
    # Extraire les co-occurrences de l'ingrédient sélectionné
    co_occurrences = co_occurrence_matrix.loc[selected_ingredient]

    # Exclure les zéros
    # Transformer les co-occurrences en un dictionnaire pour le nuage de mots
    word_frequencies = co_occurrences[co_occurrences > 0].to_dict()

    # Générer le nuage de mots
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=200
    ).generate_from_frequencies(word_frequencies)

    # Afficher le nuage de mots dans Streamlit
    st.subheader(f"Nuage de mots pour '{selected_ingredient}'")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def display_rating_ingredientbyfeature(df_results):
    """
    Affiche un scatter plot 3D interactif avec Streamlit.

    Arguments :
    - df_results : DataFrame contenant les colonnes 'minutes', 'n_steps', 'n_ingredients', et 'rating'.
    """

    # Création de la figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(
        df_results['minutes'], df_results['n_steps'], df_results['n_ingredients'],
        c=df_results['rating'], cmap='coolwarm', s=50, alpha=0.7
    )

    # Labels
    ax.set_xlabel('Minutes')
    ax.set_ylabel('N Steps')
    ax.set_zlabel('N Ingredients')
    ax.set_title('Relation entre Minutes, N Steps, N Ingredients et Rating')

    # Ajouter une barre de couleur
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Rating')

    # Afficher la figure avec Streamlit
    st.pyplot(fig)


def display_minutes_byfeature(df_results):
    """
    Affiche un scatter plot 3D interactif avec Streamlit.

    Arguments :
    - df_results : DataFrame contenant les colonnes 'minutes', 'n_steps', 'n_ingredients', et 'len_steps'.
    """

    # Création de la figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(
        df_results['n_steps'], df_results['n_ingredients'], df_results['len_steps'],
        c=df_results['minutes'], cmap='coolwarm', s=50, alpha=0.7
    )

    # Labels
    ax.set_xlabel('N Steps')
    ax.set_ylabel('N Ingredients')
    ax.set_zlabel('Len Steps')
    ax.set_title('3D Scatter Plot avec Minutes comme Intensité')

    # Ajouter une barre de couleur
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Minutes')

    # Afficher la figure avec Streamlit
    st.pyplot(fig)
