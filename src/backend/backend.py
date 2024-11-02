"""
Backend va fournir l'ensemble des méthodes permettant le calcul et la mise en forme des données
L'affichage se fera via Frontend
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
    recipe = dict()
    i = 1
    while i < 41:
        recipe[i] = len(cook.Recipe.get_all(session, n_ingredients=i))
        i += 1
    return recipe


def top_ingredient_used(session, n):
    # Requête pour compter le nombre de recettes par ingredient_id
    results = (
        session.query(
            cook.Ingredient.name,
            func.count(cook.recipe_ingredient.c.recipe_id).label(
                'recipe_count')
        )
        .join(cook.recipe_ingredient, cook.Ingredient.ingredient_id == cook.recipe_ingredient.c.ingredient_id)
        .group_by(cook.Ingredient.name)
        .all()
    )

    # Trier les résultats par nombre de recettes en ordre décroissant
    nbingredientsrecettes_trie = sorted(
        results, key=lambda x: x[1], reverse=True)

    # Récupérer les n premiers ingrédients
    return nbingredientsrecettes_trie[:n]


def top_ingredient_rating(session):
    results = (
        session.query(
            cook.Ingredient.name,
            func.count(cook.Review.review_id).label('review_count'),
            func.avg(cook.Review.rating).label('average_rating')
        )
        .join(cook.reviewer_recipe_review, cook.Review.review_id == cook.reviewer_recipe_review.c.review_id)
        .join(cook.recipe_ingredient, cook.reviewer_recipe_review.c.recipe_id == cook.recipe_ingredient.c.recipe_id)
        .join(cook.Ingredient, cook.recipe_ingredient.c.ingredient_id == cook.Ingredient.ingredient_id)
        .group_by(cook.Ingredient.name)
        .all()
    )
    ingredient_average_rating = {result.name: result.average_rating for result in sorted(
        results, key=lambda x: x.average_rating, reverse=True)}

    ingredient_review_count = {result.name: result.review_count for result in sorted(
        results, key=lambda x: x.review_count, reverse=True)}

    return ingredient_average_rating, ingredient_review_count


# Clusterisation des ingrédients
# Création d'une matrice binaire (recette x ingrédient)
def generate_kmeans_ingredient(session, nb_cluster):
    results = (
        session.query(
            cook.Recipe.recipe_id,
            # Agréger les noms d'ingrédients dans une liste
            func.array_agg(cook.Ingredient.name).label("ingredients")
        )
        .join(cook.recipe_ingredient, cook.Recipe.recipe_id == cook.recipe_ingredient.c.recipe_id)
        .join(cook.Ingredient, cook.recipe_ingredient.c.ingredient_id == cook.Ingredient.ingredient_id)
        .group_by(cook.Recipe.recipe_id)
        .all()
    )

    # Conversion des résultats en un DataFrame
    df_recipes_ingredients = pd.DataFrame(
        [{"id_recipe": result.recipe_id, "ingredients": result.ingredients}
            for result in results]
    )

    # Avec CountVectorizer car tfidf dimunue les mots les plus fréquents
    count_vectorizer = CountVectorizer()
    X_count = count_vectorizer.fit_transform(
        df_recipes_ingredients['ingredients'].apply(lambda x: ', '.join(x))
    )

    kmeans_count = KMeans(n_clusters=nb_cluster, random_state=42)
    df_recipes_ingredients['cluster_count'] = kmeans_count.fit_predict(X_count)


# Réduction de dimension avec PCA pour visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_count.toarray())
    df_recipes_ingredients['pca_x'] = X_pca[:, 0]
    df_recipes_ingredients['pca_y'] = X_pca[:, 1]

    return df_recipes_ingredients
