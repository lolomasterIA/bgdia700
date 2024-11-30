import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch
from src.frontend import frontend


@pytest.fixture
def df_recipes_ingredients():
    # Créez un DataFrame mock pour les tests
    return pd.DataFrame(
        {
            "pca_x": [1.0, 2.0, 3.0],
            "pca_y": [1.0, 2.0, 3.0],
            "cluster": [0, 1, 1],
            "recette": ["Recipe 1", "Recipe 2", "Recipe 3"],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "ingredient": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
        }
    )


@pytest.fixture
def co_occurrence_matrix():
    # Créez une matrice de co-occurrence mock pour les tests
    return pd.DataFrame(
        {
            "Ingredient 1": [0, 1, 2],
            "Ingredient 2": [1, 0, 3],
            "Ingredient 3": [2, 3, 0],
        },
        index=["Ingredient 1", "Ingredient 2", "Ingredient 3"],
    )


@pytest.fixture
def df_results():
    # Créez un DataFrame mock pour les tests
    return pd.DataFrame(
        {
            "minutes": [10, 20, 30],
            "n_steps": [5, 10, 15],
            "n_ingredients": [3, 6, 9],
            "rating": [4.5, 3.0, 5.0],
            "len_steps": [50, 100, 150],
        }
    )


@patch("src.frontend.frontend.st")  # Mocker la bibliothèque Streamlit
@patch("src.frontend.frontend.st_navbar")  # Mocker la fonction st_navbar
def test_generate_layout(mock_st_navbar, mock_st):
    # Configurer les mocks
    mock_st_navbar.return_value = "Généralité"  # Simule la sélection du menu
    mock_st.columns.return_value = ("col1", "col2")  # Simule la création des colonnes

    # Appeler la fonction
    menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()

    # Assertions pour vérifier le comportement attendu
    mock_st.set_page_config.assert_called_once_with(
        page_title="Mange ta main",
        page_icon="src/frontend/images/favicon.png",
        layout="wide",
    )
    mock_st.markdown.assert_called()  # Vérifie que le style est appliqué
    mock_st.image.assert_called_with(
        "src/frontend/images/mangetamain.jpg"
    )  # Vérifie l'image

    # Vérifie que st_navbar est appelé correctement
    mock_st_navbar.assert_called_once()
    assert menu == "Généralité"  # Vérifie le menu sélectionné
    assert col1 == "col1"  # Vérifie les colonnes retournées
    assert col2 == "col2"


@patch("src.frontend.frontend.st")  # Mocker la bibliothèque Streamlit
@patch("src.frontend.frontend.st_navbar")  # Mocker la fonction st_navbar
def test_generate_layout_with_columns(mock_st_navbar, mock_st):
    # Configurer les mocks
    mock_st_navbar.return_value = "Clusterisation"  # Simule la sélection du menu
    mock_st.columns.side_effect = lambda x: (
        tuple(f"col_{i}" for i in x)
        if isinstance(x, list)
        else tuple(f"col_{i}" for i in range(x))
    )

    # Appeler la fonction
    menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()

    # Vérifie les colonnes retournées pour le menu spécifique
    assert col1 == "col_1"
    assert col2 == "col_2"
    mock_st.columns.assert_any_call(
        [1, 2]
    )  # Vérifie que la bonne configuration a été utilisée

    # Vérifier d'autres éléments si nécessaire
    assert menu == "Clusterisation"


def test_display_kmeans_ingredient(df_recipes_ingredients):
    # Mock la fonction Streamlit plotly_chart
    with patch("streamlit.plotly_chart") as plotly_chart:
        frontend.display_kmeans_ingredient(df_recipes_ingredients)
        assert plotly_chart.called


def test_display_cluster_recipe(df_recipes_ingredients):
    # Mock la fonction Streamlit plotly_chart
    with patch("streamlit.plotly_chart") as plotly_chart:
        frontend.display_cluster_recipe(df_recipes_ingredients)
        assert plotly_chart.called


def test_display_cloud_ingredient(co_occurrence_matrix):
    # Mock la fonction Streamlit pyplot
    with patch("streamlit.pyplot") as pyplot:
        frontend.display_cloud_ingredient(co_occurrence_matrix, "Ingredient 1")
        assert pyplot.called


def test_display_rating_ingredientbyfeature(df_results):
    # Mock la fonction Streamlit pyplot
    with patch("streamlit.pyplot") as pyplot:
        frontend.display_rating_ingredientbyfeature(df_results)
        assert pyplot.called


def test_display_minutes_byfeature(df_results):
    # Mock la fonction Streamlit pyplot
    with patch("streamlit.pyplot") as pyplot:
        frontend.display_minutes_byfeature(df_results)
        assert pyplot.called
