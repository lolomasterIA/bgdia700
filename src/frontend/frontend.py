"""
Frontend va fournir l'ensemble des méthodes permettant l'affichage.
"""

import streamlit as st
from streamlit_navigation_bar import st_navbar
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_layout():
    """
    Configure et génère la mise en page principale de l'application Streamlit.

    Retourne
    --------
    tuple
        Contient le menu sélectionné et les objets de colonne pour structurer le contenu dans des sections.
    """
    # Configurer la page pour un affichage en pleine largeur
    st.set_page_config(
        page_title="Mange ta main",
        page_icon="src/frontend/images/favicon.png",  # Chemin relatif vers l'icône
        layout="wide",
    )

    # Appliquer une feuille de style via HTML/CSS
    st.markdown(
        """
    <style>
    /* Styles globaux */
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #FFF4E1; /* Fond légèrement orangé */
        color: #4A2C2A; /* Couleur de texte foncée pour un bon contraste */
    }

    /* Titres */
    h1 {
        color: #E65100; /* Orange vif */
        font-size: 2.5em;
        margin-bottom: 0.5em;
        border-bottom: 2px solid #FFB74D; /* Ligne orange clair */
        padding-bottom: 0.2em;
    }

    h2 {
        color: #F57C00; /* Orange légèrement plus clair */
        font-size: 2em;
        margin-bottom: 0.1em;
    }

    h3 {
        color: #FB8C00; /* Ton orangé intermédiaire */
        font-size: 1.5em;
        margin-bottom: 0.3em;
    }

    /* Texte de paragraphe */
    p {
        font-size: 1.1em;
        line-height: 1.6;
        margin: 1em 0;
    }
    ul {
    margin: 0;
    padding-left: 1.5em;
    list-style-type: circle;
    }
    ol {
        margin: 0;
        padding-left: 1.5em;
        list-style-type: decimal;
    }
    li {
        margin-bottom: 5px;
        margin: 0;
    }
    .main > div {
        padding-top: 0; /* Supprime l'espace entre l'image et le menu */
    }
    """,
        unsafe_allow_html=True,
    )

    # Titre principal de l'application
    st.image("src/frontend/images/mangetamain.jpg")

    stylesmenu = {
        "nav": {
            "background-color": "darkorange",
            "justify-content": "left",
        },
        "img": {
            "padding-right": "14px",
        },
        "span": {
            "color": "white",
            "padding": "14px",
        },
        "active": {
            "background-color": "white",
            "color": "darkorange",
            "font-weight": "normal",
            "padding": "14px",
        },
        "hover": {
            "background-color": "white",
            "color": "darkorange",
        },
    }

    menu = st_navbar(
        [
            "Généralité",
            "Clusterisation",
            "Ingrédients qui vont bien ensemble",
            "Corrélation rating ingrédient",
            "Corrélation minutes",
        ],
        styles=stylesmenu,
        adjust=False,
    )

    # Zone principale de contenu
    st.header(menu)

    # Utilisation des colonnes pour diviser la zone centrale en plusieurs sections
    with st.container():
        if (
            menu == "Clusterisation"
            or menu == "Ingrédients qui vont bien ensemble"
            or menu == "Corrélation rating ingrédient"
            or menu == "Corrélation minutes"
        ):
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

    Paramètres
    ----------
    df_recipes_ingredients : pd.DataFrame
        DataFrame contenant les informations de clusterisation et les coordonnées PCA.

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

    Paramètres
    ----------
    df_recipes_ingredients : pd.DataFrame
        DataFrame contenant les informations de clusterisation et les coordonnées PCA.

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
    Affiche un ingrédient avec la distance des autres ingrédients.

    Paramètres
    ----------
    co_occurrence_matrix : np.array
        Matrice de co-occurrences des ingrédients.
    selected_ingredient : str
        L'ingrédient à comparer.
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
        max_words=200,
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

    Paramètres
    ----------
    df_results : pd.DataFrame
        DataFrame contenant les colonnes 'minutes', 'n_steps', 'n_ingredients', et 'rating'.
    """

    # Création de la figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    scatter = ax.scatter(
        df_results["minutes"],
        df_results["n_steps"],
        df_results["n_ingredients"],
        c=df_results["rating"],
        cmap="coolwarm",
        s=50,
        alpha=0.7,
    )

    # Labels
    ax.set_xlabel("Minutes")
    ax.set_ylabel("N Steps")
    ax.set_zlabel("N Ingredients")
    ax.set_title("Relation entre Minutes, N Steps, N Ingredients et Rating")

    # Ajouter une barre de couleur
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Rating")

    # Afficher la figure avec Streamlit
    st.pyplot(fig)


def display_minutes_byfeature(df_results):
    """
    Affiche un scatter plot 3D interactif avec Streamlit.

    Paramètres
    ----------
    df_results : pd.DataFrame
        DataFrame contenant les colonnes 'minutes', 'n_steps', 'n_ingredients', et 'len_steps'.
    """

    # Création de la figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot
    scatter = ax.scatter(
        df_results["n_steps"],
        df_results["n_ingredients"],
        df_results["len_steps"],
        c=df_results["minutes"],
        cmap="coolwarm",
        s=50,
        alpha=0.7,
    )

    # Labels
    ax.set_xlabel("N Steps")
    ax.set_ylabel("N Ingredients")
    ax.set_zlabel("Len Steps")
    ax.set_title("3D Scatter Plot avec Minutes comme Intensité")

    # Ajouter une barre de couleur
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Minutes")

    # Afficher la figure avec Streamlit
    st.pyplot(fig)


def display_sidebar(nb_ingredient, selected_data):
    """
    Affiche les informations dans la barre latérale de Streamlit.

    Parameters
    ----------
    nb_ingredient : int
        Nombre total d'ingrédients.
    selected_data : str
        Type de données sélectionné pour les ingrédients :
        "One word" pour les ingrédients à un mot, ou un autre type pour inclure tous les ingrédients.

    Returns
    -------
    None
        Cette fonction n'a pas de retour explicite. Elle affiche du contenu dans l'interface Streamlit.
    """
    st.write(f"Nombre d'ingrédients : {nb_ingredient}")
    st.write("Ingrédients utilisés dans les analyses :")

    if selected_data == "One word":
        st.write(
            "Récupération des ingrédients à un mot. "
            "Réduction des autres ingrédients à un mot parmi ceux sélectionnés précédemment."
        )
    else:
        st.write("Tous les ingrédients du dataset initial.")
