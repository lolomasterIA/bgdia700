import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from collections import namedtuple
from src.backend import backend
from sklearn.utils._testing import assert_almost_equal


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


def test_top_ingredient_used(session, mock_data, data_type="One word"):
    # Mock la requête SQLAlchemy
    session.query().join().group_by().all.return_value = [
        ("Ingredient 1", 10),
        ("Ingredient 2", 20),
        ("Ingredient 3", 5),
    ]
    result = backend.top_ingredient_used(session, 2, data_type)
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
    avg_rating, review_count = backend.top_ingredient_rating(
        session, data_type)
    assert isinstance(avg_rating, dict)
    assert isinstance(review_count, dict)
    assert avg_rating["Ingredient 3"] == 5.0
    assert review_count["Ingredient 2"] == 20


def test_generate_cluster_recipe(session):
    # Mock des données pour simuler la base de données
    RecipeIngredients = namedtuple(
        "RecipeIngredients", ["name", "ingredients"])
    session.query().join().join().group_by().where().all.return_value = [
        RecipeIngredients("Recipe 1", ["Ingredient 1", "Ingredient 2"]),
        RecipeIngredients("Recipe 2", ["Ingredient 2", "Ingredient 3"]),
        RecipeIngredients("Recipe 3", ["Ingredient 1", "Ingredient 3"]),
    ]

    # Tester chaque combinaison de paramètres
    for matrix_type in ["tfidf", "count"]:
        for clustering_type in ["kmeans", "dbscan", "agglomerative"]:
            for reduction_type in ["pca", "svd"]:
                result_df, total_recipes, total_ingredients = (
                    backend.generate_cluster_recipe(
                        session,
                        matrix_type=matrix_type,
                        reduction_type=reduction_type,
                        clustering_type=clustering_type,
                        n_components=2,
                        nb_cluster=2,
                        data_type="One word"
                    )
                )

                # Vérifications générales
                assert isinstance(result_df, pd.DataFrame)
                assert "recette" in result_df.columns
                assert "cluster" in result_df.columns
                assert "pca_x" in result_df.columns
                assert "pca_y" in result_df.columns
                assert total_recipes == 3
                assert total_ingredients == 3  # Nombre unique d'ingrédients

                # Vérifications spécifiques pour les clusters
                if clustering_type == "kmeans":
                    assert result_df["cluster"].nunique() <= 2
                elif clustering_type == "dbscan":
                    # DBSCAN peut produire un cluster -1 pour les outliers
                    assert result_df["cluster"].nunique() > 0
                elif clustering_type == "agglomerative":
                    assert result_df["cluster"].nunique() <= 2

    # Cas : type de matrice invalide
    with pytest.raises(ValueError, match="Type de matrice non supporté."):
        backend.generate_cluster_recipe(
            session,
            matrix_type="invalid",
            reduction_type="pca",
            clustering_type="kmeans",
            n_components=2,
            nb_cluster=2,
            data_type="One word"
        )

    # Cas : type de cluster invalide
    with pytest.raises(ValueError, match="Type de clusterisation non supporté."):
        backend.generate_cluster_recipe(
            session,
            matrix_type="tfidf",
            reduction_type="pca",
            clustering_type="invalid",
            n_components=2,
            nb_cluster=2,
            data_type="One word"
        )

    # Cas : type de réduction invalide
    with pytest.raises(ValueError, match="Type de réduction non supporté."):
        backend.generate_cluster_recipe(
            session,
            matrix_type="tfidf",
            reduction_type="invalid",
            clustering_type="kmeans",
            n_components=2,
            nb_cluster=2,
            data_type="One word"
        )


def test_generate_kmeans_ingredient(session, mock_data, data_type="One word"):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    RecipeIngredients = namedtuple(
        "RecipeIngredients", ["recipe_id", "ingredients"])

    # Mock la requête SQLAlchemy
    session.query().join().join().group_by().where().all.return_value = [
        RecipeIngredients(1, ["Ingredient 1", "Ingredient 2"]),
        RecipeIngredients(2, ["Ingredient 2", "Ingredient 3"]),
    ]
    reduced_data, all_ingredients, kmeans = backend.generate_kmeans_ingredient(
        session, 2, data_type
    )
    assert isinstance(reduced_data, np.ndarray)
    assert isinstance(all_ingredients, list)
    assert len(all_ingredients) == 3
    assert kmeans.n_clusters == 2


def test_generate_matrice_ingredient(session, mock_data, data_type="One word"):
    RecipeIngredients = namedtuple(
        "RecipeIngredients", ["recipe_id", "ingredients"])

    # Mock la requête SQLAlchemy
    session.query().join().join().group_by().where().all.return_value = [
        RecipeIngredients(1, ["Ingredient 1", "Ingredient 2"]),
        RecipeIngredients(2, ["Ingredient 2", "Ingredient 3"]),
    ]

    co_occurrence_matrix, all_ingredients = backend.generate_matrice_ingredient(
        session, data_type)

    assert isinstance(co_occurrence_matrix, pd.DataFrame)
    assert isinstance(all_ingredients, list)
    assert "Ingredient 1" in co_occurrence_matrix.columns
    assert "Ingredient 2" in co_occurrence_matrix.index


def test_suggestingredients(session, mock_data):
    # Créer une matrice de co-occurrence simulée
    co_occurrence_matrix = pd.DataFrame(
        {
            "Ingredient 1": {"Ingredient 1": 0, "Ingredient 2": 5, "Ingredient 3": 3},
            "Ingredient 2": {"Ingredient 1": 5, "Ingredient 2": 0, "Ingredient 3": 8},
            "Ingredient 3": {"Ingredient 1": 3, "Ingredient 2": 8, "Ingredient 3": 0},
        }
    )
    result = backend.suggestingredients(
        co_occurrence_matrix, "Ingredient 2", top_n=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert (
        result[0][0] == "Ingredient 3"
    )  # Vérifier que l'ingrédient suggéré est correct
    assert result[0][1] == 8  # Vérifier le score


def test_get_ingredient_rating(session, data_type="One word"):
    # Mock la requête SQLAlchemy
    session.query().filter().first.return_value = namedtuple(
        "Rating", ["rating"])(4.5)

    result = backend.get_ingredient_rating(session, "Ingredient 1", data_type)
    assert isinstance(result, float)
    assert result == 4.5


def test_generate_regression_ingredient(session):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    RecipeIngredients = namedtuple(
        "RecipeIngredients",
        [
            "ingredient_id",
            "recipe_id",
            "minutes",
            "n_steps",
            "sum_rating",
            "n_ingredients",
            "len_steps",
            "len_description",
            "rating",
        ],
    )

    # Mock la requête SQLAlchemy
    session.query().join().join().filter().all.return_value = [
        RecipeIngredients(1, 1, 30, 5, 4.5, 4, 100, 200, 4.5),
        RecipeIngredients(2, 2, 45, 6, 4.0, 5, 150, 250, 4.0),
        RecipeIngredients(3, 3, 60, 7, 3.5, 6, 200, 300, 3.5),
    ]

    # Tester chaque modèle
    for model_type in ["rl", "xgb", "rf"]:
        mse, r2, coefficients, df_results = backend.generate_regression_ingredient(
            session, model=model_type
        )

        # Vérifiez les résultats
        assert isinstance(mse, float)
        assert isinstance(r2, float)
        assert isinstance(df_results, pd.DataFrame)
        if model_type == "rl":
            assert isinstance(coefficients, pd.DataFrame)
        else:
            assert coefficients is None

        # Vérifiez que le DataFrame contient les bonnes colonnes
        expected_columns = [
            "ingredient_id",
            "recipe_id",
            "minutes",
            "n_steps",
            "sum_rating",
            "n_ingredients",
            "len_steps",
            "len_description",
            "rating",
        ]
        assert list(df_results.columns) == expected_columns

        # Vérifiez que le DataFrame contient les bonnes données
        assert len(df_results) == 3
        assert df_results["ingredient_id"].tolist() == [1, 2, 3]


def test_generate_regression_minutes(session):
    # Définir un namedtuple pour simuler les résultats de la requête SQLAlchemy
    RecipeData = namedtuple(
        "RecipeData",
        [
            "recipe_id",
            "minutes",
            "n_steps",
            "n_ingredients",
            "len_steps",
            "len_description",
        ],
    )

    # Mock la requête SQLAlchemy
    session.query().all.return_value = [
        RecipeData(1, 30, 5, 4, 100, 200),
        RecipeData(2, 45, 6, 5, 150, 250),
        RecipeData(3, 60, 7, 6, 200, 300),
    ]

    # Tester chaque modèle
    for model_type in ["rl", "xgb", "rf"]:
        mse, r2, coefficients, df_results = backend.generate_regression_minutes(
            session, model=model_type
        )

        # Vérifiez les résultats
        assert isinstance(mse, float)
        assert isinstance(r2, float)
        assert isinstance(df_results, pd.DataFrame)
        if model_type == "rl":
            assert isinstance(coefficients, pd.DataFrame)
        else:
            assert coefficients is None

        # Vérifiez que le DataFrame contient les bonnes colonnes
        expected_columns = [
            "recipe_id",
            "minutes",
            "n_steps",
            "n_ingredients",
            "len_steps",
            "len_description",
            "predicted_minutes",
        ]
        assert list(df_results.columns) == expected_columns

        # Vérifiez que le DataFrame contient les bonnes données
        assert len(df_results) == 3
        assert df_results["recipe_id"].tolist() == [1, 2, 3]

    # Tester un modèle non reconnu
    with pytest.raises(
        ValueError, match="Modèle non reconnu : choisissez 'rl', 'xgb' ou 'rf'."
    ):
        backend.generate_regression_minutes(session, model="unknown_model")


def test_delete_outliers():
    # Créer un DataFrame de test
    data = {
        "minutes": [10, 20, 30, 40, 50, 1000],
        "n_steps": [1, 2, 3, 4, 5, 6],
        "n_ingredients": [1, 2, 3, 4, 5, 6],
    }
    df = pd.DataFrame(data)

    # Tester la méthode "DeleteQ1Q3"
    df_reduced = backend.delete_outliers(
        df.copy(), key="minutes", method="DeleteQ1Q3")
    assert len(df_reduced) == 5
    assert 1000 not in df_reduced["minutes"].values

    # Tester la méthode "Capping"
    df_reduced = backend.delete_outliers(
        df.copy(), key="minutes", method="Capping")
    assert len(df_reduced) == 6
    assert 1000 not in df_reduced["minutes"].values

    # Tester la méthode "Log"
    df_reduced = backend.delete_outliers(
        df.copy(), key="minutes", method="Log")
    assert np.allclose(df_reduced["minutes"], np.log1p(df["minutes"]))

    # Tester la méthode "Isolation Forest" avec des paramètres ajustés
    df_reduced = backend.delete_outliers(
        df.copy(), key="minutes", method="Isolation Forest", contamination=0.2
    )
    assert len(df_reduced) == 5  # Une entrée doit être supprimée
    assert 1000 not in df_reduced["minutes"].values

    # Tester la méthode "DBScan" avec des paramètres ajustés
    df_reduced = backend.delete_outliers(
        df.copy(), key="minutes", method="DBScan", eps=100, min_samples=2
    )
    assert len(df_reduced) == 5  # Une entrée doit être supprimée
    assert 1000 not in df_reduced["minutes"].values

    # Tester la méthode "Local Outlier Factor" avec des paramètres ajustés
    df_reduced = backend.delete_outliers(
        df.copy(),
        key="minutes",
        method="Local Outlier Factor",
        n_neighbors=2,
        contamination=0.2,
    )
    print(df_reduced)  # Ajoutez cette ligne pour voir le DataFrame résultant
    assert len(df_reduced) == 5  # Une entrée doit être supprimée
    assert 1000 not in df_reduced["minutes"].values

    # Test for an unknown method
    with pytest.raises(ValueError):
        backend.delete_outliers(df.copy(), key="minutes",
                                method="Unknown Method")
