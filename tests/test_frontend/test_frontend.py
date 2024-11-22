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


def test_generate_layout_generalite():
    # Mock les fonctions Streamlit pour le menu "Généralité"
    with patch("streamlit.set_page_config"), patch("streamlit.image"), patch(
        "streamlit.selectbox", return_value="Généralité"
    ), patch("streamlit.header"), patch("streamlit.container"), patch(
        "streamlit.columns", return_value=(None, None)
    ), patch(
        "streamlit.markdown"
    ), patch(
        "streamlit.text"
    ):
        menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()
        assert menu == "Généralité"
        assert col1 is None
        assert col2 is None
        assert col3 is None
        assert col4 is None
        assert col5 is None
        assert col6 is None


def test_generate_layout_clusterisation():
    # Mock les fonctions Streamlit pour le menu "Clusterisation"
    with patch("streamlit.set_page_config"), patch("streamlit.image"), patch(
        "streamlit.selectbox", return_value="Clusterisation"
    ), patch("streamlit.header"), patch("streamlit.container"), patch(
        "streamlit.columns", return_value=(None, None)
    ), patch(
        "streamlit.markdown"
    ), patch(
        "streamlit.text"
    ):
        menu, col1, col2, col3, col4, col5, col6 = frontend.generate_layout()
        assert menu == "Clusterisation"
        assert col1 is None
        assert col2 is None
        assert col3 is None
        assert col4 is None
        assert col5 is None
        assert col6 is None


def test_display_kmeans_recipe(df_recipes_ingredients):
    # Mock la fonction Streamlit plotly_chart
    with patch("streamlit.plotly_chart") as plotly_chart:
        frontend.display_kmeans_recipe(df_recipes_ingredients)
        assert plotly_chart.called


def test_display_kmeans_ingredient(df_recipes_ingredients):
    # Mock la fonction Streamlit plotly_chart
    with patch("streamlit.plotly_chart") as plotly_chart:
        frontend.display_kmeans_ingredient(df_recipes_ingredients)
        assert plotly_chart.called
