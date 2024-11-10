"""
Backend va fournir l'ensemble des méthodes permettant le calcul et la mise en forme des données.

L'affichage se fera via Frontend.
"""

import src.backend.datalayer.cooking as cook
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sqlalchemy import func


### Travaux sur les ingrédients ###
def recipe_number_ingredient(session):
    """
    Get the number of recipes for each number of ingredients from 1 to 40.

    Parameters
    ----------
    session : Session
        The database session used to query the recipes.

    Returns
    -------
    dict
        A dictionary where the keys are the number of ingredients (1 to 40)
        and the values are the number of recipes with that number of ingredients.
    """
    recipe = dict()
    i = 1
    while i < 41:
        recipe[i] = len(cook.Recipe.get_all(session, n_ingredients=i))
        i += 1
    return recipe


def top_ingredient_used(session, n):
    """
    Get the top n ingredients used in recipes.

    Parameters
    ----------
    session : Session
        The database session used to query the ingredients.
    n : int
        The number of top ingredients to return.

    Returns
    -------
    list
        A list of tuples where each tuple contains the ingredient name and the
        number of recipes that use that ingredient, sorted in descending order
        by the number of recipes.
    """
    # Requête pour compter le nombre de recettes par ingredient_id
    results = (
        session.query(
            cook.Ingredient.name,
            func.count(cook.recipe_ingredient.c.recipe_id).label("recipe_count"),
        )
        .join(
            cook.recipe_ingredient,
            cook.Ingredient.ingredient_id == cook.recipe_ingredient.c.ingredient_id,
        )
        .group_by(cook.Ingredient.name)
        .all()
    )

    # Trier les résultats par nombre de recettes en ordre décroissant
    nbingredientsrecettes_trie = sorted(results, key=lambda x: x[1], reverse=True)

    # Récupérer les n premiers ingrédients
    return nbingredientsrecettes_trie[:n]


def top_ingredient_rating(session):
    """
    Get the average rating and review count for each ingredient.

    Parameters
    ----------
    session : Session
        The database session used to query the ingredients and reviews.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - ingredient_average_rating (dict): A dictionary where the keys are ingredient names
          and the values are the average ratings, sorted in descending order by average rating.
        - ingredient_review_count (dict): A dictionary where the keys are ingredient names
          and the values are the review counts, sorted in descending order by review count.
    """
    results = (
        session.query(
            cook.Ingredient.name,
            func.count(cook.Review.review_id).label("review_count"),
            func.avg(cook.Review.rating).label("average_rating"),
        )
        .join(
            cook.reviewer_recipe_review,
            cook.Review.review_id == cook.reviewer_recipe_review.c.review_id,
        )
        .join(
            cook.recipe_ingredient,
            cook.reviewer_recipe_review.c.recipe_id
            == cook.recipe_ingredient.c.recipe_id,
        )
        .join(
            cook.Ingredient,
            cook.recipe_ingredient.c.ingredient_id == cook.Ingredient.ingredient_id,
        )
        .group_by(cook.Ingredient.name)
        .all()
    )
    ingredient_average_rating = {
        result.name: result.average_rating
        for result in sorted(results, key=lambda x: x.average_rating, reverse=True)
    }

    ingredient_review_count = {
        result.name: result.review_count
        for result in sorted(results, key=lambda x: x.review_count, reverse=True)
    }

    return ingredient_average_rating, ingredient_review_count


# Clusterisation des ingrédients
# Création d'une matrice binaire (recette x ingrédient)
def generate_kmeans_ingredient(session, nb_cluster):
    """
    Generate K-means clusters based on ingredients in recipes.

    This function reduces the dataset by removing:
    - Recipes with less than 3 ingredients.
    - Recipes with less than 20 reviews.
    - Ingredients that appear in less than 5 recipes.
    - Associated ingredients/recipes.

    Parameters
    ----------
    session : Session
        The database session used to query the recipes and ingredients.
    nb_cluster : int
        The number of clusters to generate.

    Returns
    -------
    list
        A list of results containing recipe IDs and aggregated ingredient names.
    """
    # réduction du dataset, on enlève :
    # les recettes avec moins de 3 ingrédientss
    # les recettes avec moins de 20 reviews
    # les ingrédients qui apparaissent dans moins de 5 recettes
    # et les ingrédients / recettes associés
    results = (
        session.query(
            cook.Recipe.recipe_id,
            # Agréger les noms d'ingrédients dans une liste
            func.array_agg(cook.Ingredient.name).label("ingredients"),
        )
        .join(
            cook.recipe_ingredient,
            cook.Recipe.recipe_id == cook.recipe_ingredient.c.recipe_id,
        )
        .join(
            cook.Ingredient,
            cook.recipe_ingredient.c.ingredient_id == cook.Ingredient.ingredient_id,
        )
        .group_by(cook.Recipe.recipe_id)
        .where(
            (cook.Recipe.nb_rating > 20)
            & (cook.Recipe.n_ingredients > 3)
            & (cook.Ingredient.nb_recette > 5)
        )
        .all()
    )

    # Conversion des résultats en un DataFrame
    df_recipes_ingredients = pd.DataFrame(
        [
            {"id_recipe": result.recipe_id, "ingredients": result.ingredients}
            for result in results
        ]
    )

    print(df_recipes_ingredients["ingredients"])
    print(len(df_recipes_ingredients["ingredients"]))
    # Avec CountVectorizer car tfidf dimunue les mots les plus fréquents
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "))
    # Transpose pour avoir ingrédients en lignes
    X_count = vectorizer.fit_transform(
        df_recipes_ingredients["ingredients"].apply(lambda x: ", ".join(x))
    ).T
    # Obtenir les noms des ingrédients
    ingredient_names = vectorizer.get_feature_names_out()

    # Clusterisation avec KMeans des ingrédients (chaque ligne est un ingrédient)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_count)

    # Création d'un DataFrame pour stocker les résultats de clusterisation
    df_ingredients_clusters = pd.DataFrame(
        {"ingredient": ingredient_names, "cluster": clusters}
    )

    # Réduction de dimension avec PCA pour visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_count.toarray())
    df_ingredients_clusters["pca_x"] = X_pca[:, 0]
    df_ingredients_clusters["pca_y"] = X_pca[:, 1]

    return df_ingredients_clusters
