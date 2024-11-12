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
    Compte le nombre de recettes contenant un nombre spécifique d'ingrédients.

    Paramètres:
    - session : SQLAlchemy session pour la base de données.
    Retourne:
    - dict : Dictionnaire contenant le nombre de recettes pour chaque nombre d'ingrédients (jusqu'à 40).
    """
    recipe = dict()
    i = 1
    while i < 41:
        recipe[i] = len(cook.Recipe.get_all(session, n_ingredients=i))
        i += 1
    return recipe


def top_ingredient_used(session, n):
    """
    Récupère les n ingrédients les plus utilisés dans les recettes.

    Paramètres:
    - session : SQLAlchemy session pour la base de données.
    - n : Nombre d'ingrédients les plus utilisés à retourner.

    Retourne:
    - list : Liste des n ingrédients les plus utilisés avec leur nombre de recettes.
    """
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
    Récupère les ingrédients avec les meilleures notes moyennes et le plus de reviews.

    Paramètres:
    - session : SQLAlchemy session pour la base de données.

    Retourne:
    - tuple : Deux dictionnaires, un pour les notes moyennes et un pour le nombre de reviews par ingrédient.
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
def generate_kmeans_recipe(session, nb_cluster):
    """
    Effectue une clusterisation des ingrédients en fonction de leur utilisation dans les recettes.

    réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 20 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés.

    Paramètres:
    - session : SQLAlchemy session pour la base de données.
    - nb_cluster : Nombre de clusters pour la clusterisation.

    Retourne:
    - pd.DataFrame : DataFrame contenant les recettes, les clusters, et les coordonnées PCA pour visualisation.
    - le nombre de recette
    - le nombre d'ingrédient
    """
    results = (
        session.query(
            cook.Recipe.name,
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
            {"id_recipe": result.name, "ingredients": result.ingredients}
            for result in results
        ]
    )

    # Nombre total de recettes
    nombre_total_recettes = df_recipes_ingredients["id_recipe"].nunique()
    # Nombre total d'ingrédients (en considérant les ingrédients uniques dans toutes les recettes)
    nombre_total_ingredients = df_recipes_ingredients["ingredients"].explode().nunique()

    # Avec CountVectorizer car tfidf dimunue les mots les plus fréquents
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "))
    # Les recettes en lignes
    X_count = vectorizer.fit_transform(
        df_recipes_ingredients["ingredients"].apply(lambda x: ", ".join(x))
    )
    # Obtenir les noms des ingrédients
    # ingredient_names = vectorizer.get_feature_names_out()
    # Clusterisation avec KMeans des ingrédients (chaque ligne est un ingrédient)
    kmeans = KMeans(nb_cluster, random_state=42)
    clusters = kmeans.fit_predict(X_count)

    # Création d'un DataFrame pour stocker les résultats de clusterisation
    df_ingredients_clusters = pd.DataFrame(
        {"recette": df_recipes_ingredients["id_recipe"], "cluster": clusters}
    )

    # Réduction de dimension avec PCA pour visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_count.toarray())
    df_ingredients_clusters["pca_x"] = X_pca[:, 0]
    df_ingredients_clusters["pca_y"] = X_pca[:, 1]

    return df_ingredients_clusters, nombre_total_recettes, nombre_total_ingredients


# Création d'une matrice de co occurrences (ingrédient x ingrédient)


def generate_kmeans_ingredient(session, nb_cluster):
    """
    Effectue une clusterisation des ingrédients en fonction de leur utilisation dans les recettes.

    réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 20 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés.

    Paramètres:
    - session : SQLAlchemy session pour la base de données.
    - nb_cluster : Nombre de clusters pour la clusterisation.

    Retourne:
    - pd.DataFrame : DataFrame contenant les ingrédients, les clusters, et les coordonnées PCA pour visualisation.
    - le nombre de recette
    - le nombre d'ingrédient
    """
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

    # Extraction de tous les ingrédients uniques
    all_ingredients = list(
        set(
            ingredient
            for ingredients_list in df_recipes_ingredients["ingredients"]
            for ingredient in ingredients_list
        )
    )

    # Initialisation de la matrice de co-occurrence
    co_occurrence_matrix = pd.DataFrame(
        0, index=all_ingredients, columns=all_ingredients
    )

    # Remplissage de la matrice en comptant les co-occurrences
    for ingredients_list in df_recipes_ingredients["ingredients"]:
        for i in range(len(ingredients_list)):
            for j in range(i + 1, len(ingredients_list)):
                co_occurrence_matrix.loc[ingredients_list[i], ingredients_list[j]] += 1
                co_occurrence_matrix.loc[ingredients_list[j], ingredients_list[i]] += 1

    # Clustering avec KMeans basé sur la similarité cosinus
    similarity_matrix = cosine_similarity(co_occurrence_matrix)
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(similarity_matrix)

    # Visualisation des clusters après réduction de dimension
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(similarity_matrix)

    return reduced_data, all_ingredients, kmeans
