"""
Frontend va fournir l'ensemble des méthodes permettant l'affichage.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


def generate_layout():
    """
    Generate the layout for the Streamlit application.

    This function configures the page layout, sets the page title and icon,
    displays the main title image, and creates the main content area with
    multiple sections.

    Returns
    -------
    tuple
        A tuple containing the selected menu option and the columns for the layout.
    """
    # Configurer la page pour un affichage en pleine largeur
    st.set_page_config(
        page_title="Mange ta main",
        page_icon="src/frontend/images/favicon.png",  # Chemin relatif vers l'icône
        layout="wide",
    )
    # Titre principal de l'application
    st.image("src/frontend/images/mangetamain.jpg", use_container_width=True)

    menu = st.selectbox("", ["Généralité", "Clusterisation", "Page 3"])

    # Zone principale de contenu
    st.header(menu)

    # Utilisation des colonnes pour diviser la zone centrale en plusieurs sections
    with st.container():
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
def display_kmeans_ingredient(df_recipes_ingredients):
    """
    Display the K-means clusters of ingredients in recipes.

    This function creates a scatter plot of the K-means clusters using Plotly
    and displays it in the Streamlit application.

    Parameters
    ----------
    df_recipes_ingredients : pd.DataFrame
        A DataFrame containing the PCA components and cluster assignments for the ingredients.
    """
    # Affichage des clusters sous forme de graphique
    fig = px.scatter(
        df_recipes_ingredients,
        x="pca_x",
        y="pca_y",
        color="cluster",
        hover_data=["ingredient"],
        title="Cluster des ingrédients en fonction des recettes",
    )
    st.plotly_chart(fig)


"""
    # Visualisation avec Matplotlib
    plt.figure(figsize=(10, 6))
    for cluster in df_recipes_ingredients['cluster_count'].unique():
        cluster_data = df_recipes_ingredients[df_recipes_ingredients['cluster_count'] == cluster]
        plt.scatter(cluster_data['pca_x'],
                    cluster_data['pca_y'], label=f'Cluster {cluster}')

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Cluster des recettes en fonction des ingrédients")
    plt.legend()
    st.pyplot(plt)"""
