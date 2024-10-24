"""
Backend va fournir l'ensemble des méthodes permettant le calcul et la mise en forme des données
L'affichage se fera via Frontend
"""

import src.backend.datalayer.cooking as cook
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA


### Travaux sur les ingrédients ###
# Clusterisation des ingrédients
# Création d'une matrice binaire (recette x ingrédient)
def generate_kmeans_ingredient(nb_cluster, recipes, ingredients):
    ingredient_matrix = np.zeros((len(recipes), len(ingredients)), dtype=bool)
    asso_idnumpy_idrecipe = np.zeros(len(recipes), dtype=bool)
    asso_idnumpy_idingredient = np.zeros(len(ingredients), dtype=bool)
    for r, recette in enumerate(recipes):
        for i, ingredient in enumerate(ingredients):
            if ingredient in recette.ingredients:
                ingredient_matrix[r, i] = 1
            if i == 0:
                asso_idnumpy_idingredient[i] = ingredient.ingredient_id
        asso_idnumpy_idrecipe[r] = recette.recipe_id

    # Choisir 500 clusters par exemple
    kmeans = MiniBatchKMeans(n_clusters=nb_cluster,
                             batch_size=1000, random_state=42)
    # On transpose pour regrouper par ingrédients, pas par recette
    kmeans.fit(ingredient_matrix.T)

    # Assignation des clusters aux ingrédients
    labels = kmeans.labels_

    # Réduire les ingrédients selon les clusters
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)

    print(f"Nombre de clusters : {len(unique_clusters)}")
    # = pd.DataFrame(unique_clusters, cluster_counts=np.unique(
    #    labels, return_counts=True), columns=['PC1', 'PC2'])
    # df['Cluster'] = labels
    # df['Ingrédient'] = ingredients

    return None
