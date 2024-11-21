import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from collections import namedtuple
from src.backend import backend


@pytest.fixture
def session():
    # Créez un mock de la session SQLAlchemy
    return MagicMock()


@pytest.fixture
def mock_data():
    # Créez des données mock pour les tests
    return {
        "recipes": [
            {"name": "Recipe 1", "n_ingredients": 5, "nb_rating": 25},
            {"name": "Recipe 2", "n_ingredients": 3, "nb_rating": 30},
            {"name": "Recipe 3", "n_ingredients": 7, "nb_rating": 15},
        ],
        "ingredients": [
            {"name": "Ingredient 1", "nb_recette": 10},
            {"name": "Ingredient 2", "nb_recette": 20},
            {"name": "Ingredient 3", "nb_recette": 5},
        ],
        "reviews": [
            {"review_id": 1, "rating": 4.5},
            {"review_id": 2, "rating": 3.0},
            {"review_id": 3, "rating": 5.0},
        ],
    }


def test_recipe_number_ingredient(session, mock_data):
    # Mock la méthode get_all de Recipe
    backend.cook.Recipe.get_all = MagicMock(return_value=mock_data["recipes"])
    result = backend.recipe_number_ingredient(session)
    assert isinstance(result, dict)
    assert len(result) == 40


def test_top_ingredient_used(session, mock_data):
    # Mock la requête SQLAlchemy
    session.query().join().group_by().all.return_value = [
        ("Ingredient 1", 10),
        ("Ingredient 2", 20),
        ("Ingredient 3", 5),
    ]
    result = backend.top_ingredient_used(session, 2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0][0] == "Ingredient 2"


def test_top_ingredient_rating(session, mock_data):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    IngredientRating = namedtuple(
        "IngredientRating", ["name", "review_count", "average_rating"]
    )

    # Mock la requête SQLAlchemy
    session.query().join().join().join().group_by().all.return_value = [
        IngredientRating("Ingredient 1", 10, 4.5),
        IngredientRating("Ingredient 2", 20, 3.0),
        IngredientRating("Ingredient 3", 5, 5.0),
    ]
    avg_rating, review_count = backend.top_ingredient_rating(session)
    assert isinstance(avg_rating, dict)
    assert isinstance(review_count, dict)
    assert avg_rating["Ingredient 3"] == 5.0
    assert review_count["Ingredient 2"] == 20


def test_generate_kmeans_recipe(session, mock_data):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    RecipeIngredients = namedtuple("RecipeIngredients", ["name", "ingredients"])

    # Mock la requête SQLAlchemy
    session.query().join().join().group_by().where().all.return_value = [
        RecipeIngredients("Recipe 1", ["Ingredient 1", "Ingredient 2"]),
        RecipeIngredients("Recipe 2", ["Ingredient 2", "Ingredient 3"]),
    ]
    df, num_recipes, num_ingredients = backend.generate_kmeans_recipe(session, 2)
    assert isinstance(df, pd.DataFrame)
    assert num_recipes == 2
    assert num_ingredients == 3


def test_generate_kmeans_ingredient(session, mock_data):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    RecipeIngredients = namedtuple("RecipeIngredients", ["recipe_id", "ingredients"])

    # Mock la requête SQLAlchemy
    session.query().join().join().group_by().where().all.return_value = [
        RecipeIngredients(1, ["Ingredient 1", "Ingredient 2"]),
        RecipeIngredients(2, ["Ingredient 2", "Ingredient 3"]),
    ]
    reduced_data, all_ingredients, kmeans = backend.generate_kmeans_ingredient(
        session, 2
    )
    assert isinstance(reduced_data, np.ndarray)
    assert isinstance(all_ingredients, list)
    assert len(all_ingredients) == 3
    assert kmeans.n_clusters == 2
