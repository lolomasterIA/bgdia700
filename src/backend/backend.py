"""
Backend va fournir l'ensemble des méthodes permettant le calcul et la mise en forme des données.

L'affichage se fera via Frontend.
"""

import src.backend.datalayer.cooking as cook
import pandas as pd
import numpy as np
from sqlalchemy import func, cast, Float
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor


### Travaux sur les ingrédients ###
def recipe_number_ingredient(session):
    """
    Compte le nombre de recettes contenant un nombre spécifique d'ingrédients.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.

    Retourne
    -------
    dict
        Un dictionnaire contenant le nombre de recettes pour chaque nombre d'ingrédients (jusqu'à 40).
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

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.
    n : int
        Nombre d'ingrédients les plus utilisés à retourner.

    Retourne
    --------
    list
        Liste des n ingrédients les plus utilisés avec leur nombre de recettes.
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

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.

    Retourne
    --------
    tuple
        Deux dictionnaires, un pour les notes moyennes et un pour le nombre de reviews par ingrédient.
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
def generate_cluster_recipe(
    session,
    matrix_type="tfidf",
    reduction_type="pca",
    clustering_type="kmeans",
    n_components=2,
    nb_cluster=2,
):
    """
    Effectue une clusterisation des ingrédients en fonction de leur utilisation dans les recettes.

    Réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 15 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.
    matrix_type : str
        Type de matrice à utiliser ("tfidf", "count").
    reduction_type : str
        Réduction de dimensionnalité à appliquer ("pca", "svd").
    clustering_type : str
        Algorithme de clusterisation à utiliser ("kmeans", "dbscan", "agglomerative").
    n_components : int
        Nombre de dimensions pour la réduction.
    nb_cluster : int
        Nombre de clusters (pour KMeans ou AgglomerativeClustering).

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les recettes, les clusters, et les coordonnées PCA pour visualisation.
    int
        Le nombre de recettes.
    int
        Le nombre d'ingrédients.
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
            (cook.Recipe.nb_rating > 15)
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

    # Avec CountVectorizer et tfidf
    if matrix_type == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "))
    elif matrix_type == "count":
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "))
    else:
        raise ValueError("Type de matrice non supporté.")

    # Les recettes en lignes
    X_count = vectorizer.fit_transform(
        df_recipes_ingredients["ingredients"].apply(lambda x: ", ".join(x))
    )

    # Clusterisation
    if clustering_type == "kmeans":
        cluster_model = KMeans(n_clusters=nb_cluster, random_state=42)
    elif clustering_type == "dbscan":
        cluster_model = DBSCAN()
    elif clustering_type == "agglomerative":
        cluster_model = AgglomerativeClustering(n_clusters=nb_cluster)
    else:
        raise ValueError("Type de clusterisation non supporté.")
    clusters = cluster_model.fit_predict(X_count.toarray())

    # Création d'un DataFrame pour stocker les résultats de clusterisation
    df_ingredients_clusters = pd.DataFrame(
        {"recette": df_recipes_ingredients["id_recipe"], "cluster": clusters}
    )

    # Réduction de dimension avec PCA pour visualisation
    if reduction_type == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    elif reduction_type == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Type de réduction non supporté.")

    X_reducer = reducer.fit_transform(X_count.toarray())
    df_ingredients_clusters["pca_x"] = X_reducer[:, 0]
    df_ingredients_clusters["pca_y"] = X_reducer[:, 1]

    return df_ingredients_clusters, nombre_total_recettes, nombre_total_ingredients


# Création d'une matrice de co occurrences (ingrédient x ingrédient)


def generate_kmeans_ingredient(session, nb_cluster):
    """
    Effectue une clusterisation des ingrédients en fonction de leur utilisation dans les recettes.

    Réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 20 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.
    nb_cluster : int
        Nombre de clusters pour la clusterisation.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant les ingrédients, les clusters, et les coordonnées PCA pour visualisation.
    int
        Le nombre de recettes.
    int
        Le nombre d'ingrédients.
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


def generate_matrice_ingredient(session):
    """
    Renvoie une matrice de co-occurence des ingrédients dans les recettes.

    Réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 20 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.

    Retourne
    --------
    pd.DataFrame
        DataFrame contenant la matrice de co-occurence.
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
            (cook.Recipe.nb_rating > 15)
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
    all_ingredients = sorted(
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
    return co_occurrence_matrix, all_ingredients


def suggestingredients(co_occurrence_matrix, ingredient, top_n=5):
    """
    Suggère des ingrédients qui vont bien avec l'ingrédient donné.

    Paramètres
    ----------
    ingredient : str
        Nom de l'ingrédient sélectionné.
    top_n : int
        Nombre de suggestions à retourner.

    Retourne
    --------
    list
        Liste des meilleurs ingrédients suggérés.
    """

    # Récupère les occurrences associées à l'ingrédient
    occurrences = co_occurrence_matrix.loc[ingredient]

    # Trie les ingrédients par occurrence décroissante
    suggested = occurrences.sort_values(ascending=False).head(top_n)

    return [(index, value) for index, value in suggested[suggested > 0].items()]


def get_ingredient_rating(session, ingredient_name):
    """
    Récupère la note moyenne (rating) d'un ingrédient en fonction de sum_rating et count_review.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy active.
    ingredient_id : int
        L'identifiant de l'ingrédient.

    Retourne
    --------
    float
        La note moyenne de l'ingrédient, ou None si les données ne sont pas disponibles.
    """
    result = (
        session.query(
            (
                cast(func.sum(cook.Ingredient.sum_rating), Float)
                / cast(func.sum(cook.Ingredient.count_review), Float)
            ).label("rating")
        )
        .filter(cook.Ingredient.name == ingredient_name)
        .first()
    )

    # Retourne la note moyenne si elle existe
    return round(result.rating, 2) if result and result.rating is not None else None


# on dit que les ingrédients sont naturellement bons
# on dit que s'ils sont mauvais c'est qu'ils sont mal préparer, que la recette est compliqué
# On va voir s'il y a une corrélation avec ingredient.rating et
# recipe.minutes
# recipe.n_steps
# recipe.n_ingredients
# len(recipe.steps)
# len(description)
# Sachant qu'un ingredient intervient dans plusieurs recette


def generate_regression_ingredient(session, model="rl"):
    """
    Renvoie les résultats d'une régression linéaire sur le rating des ingrédients en fonction de la complexité de la recette.

    Réduction du dataset, on enlève :
    - les recettes avec moins de 3 ingrédients,
    - les recettes avec moins de 20 reviews,
    - les ingrédients qui apparaissent dans moins de 5 recettes et les ingrédients / recettes associés,
    - les ingrédients avec minutes > 600.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.
    model : str
        Type de modèle à utiliser ('rl', 'xgb' ou 'rf').

    Retourne
    --------
    tuple
        mse, r2 et les coefficients (si existants) du modèle.
    """
    results = (
        session.query(
            cook.Ingredient.ingredient_id,
            cook.Recipe.recipe_id,
            cook.Recipe.minutes,
            cook.Recipe.n_steps,
            cook.Ingredient.sum_rating,
            cook.Recipe.n_ingredients,
            func.char_length(cook.Recipe.steps).label("len_steps"),
            func.char_length(cook.Recipe.description).label("len_description"),
            (cook.Ingredient.sum_rating / cook.Ingredient.count_review).label("rating"),
        )
        .join(
            cook.recipe_ingredient,
            cook.Recipe.recipe_id == cook.recipe_ingredient.c.recipe_id,
        )
        .join(
            cook.Ingredient,
            cook.recipe_ingredient.c.ingredient_id == cook.Ingredient.ingredient_id,
        )
        .filter(
            cook.Recipe.n_ingredients > 3,
            cook.Ingredient.count_review > 15,
            cook.Ingredient.nb_recette > 5,
            cook.Recipe.minutes < 600,
            cook.Recipe.minutes < 600,
            (cook.Ingredient.sum_rating / cook.Ingredient.count_review) > 4,
        )
        .all()
    )
    df_results = pd.DataFrame(
        results,
        columns=[
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
    features = ["minutes", "n_steps", "n_ingredients", "len_steps", "len_description"]
    target = "rating"

    # Préparation des données
    X = df_results[features]
    y = df_results[target]

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # modèle régression linéaire
    if model == "rl":
        # Modèle de régression linéaire
        model = LinearRegression()

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Évaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Affichage des coefficients
        coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})

    # modèle Gradient Boosting
    if model == "xgb":
        # Modèle de gradient boosting
        xgb_model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)

        # Prédictions
        y_pred_xgb = xgb_model.predict(X_test)

        # Évaluation
        mse = mean_squared_error(y_test, y_pred_xgb)
        r2 = r2_score(y_test, y_pred_xgb)
        coefficients = None

    # modèle radom forest
    if model == "rf":
        # Modèle de forêt aléatoire
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)

        # Prédictions
        y_pred_rf = rf_model.predict(X_test)

        # Évaluation
        mse = mean_squared_error(y_test, y_pred_rf)
        r2 = r2_score(y_test, y_pred_rf)
        coefficients = None

    return mse, r2, coefficients, df_results


def generate_regression_minutes(session, model="rl", selected_method="DeleteQ1Q3"):
    """
    Renvoie les résultats d'une régression sur le temps (minutes) en fonction de la complexité des recettes.

    Réduction du dataset :
    - Recettes avec moins de 3 ingrédients
    - Recettes avec moins de 20 reviews
    - Ingrédients apparaissant dans moins de 5 recettes
    - Recettes avec un temps > 600 minutes

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour la base de données.
    model : str
        Type de modèle à utiliser ("rl", "xgb", "rf").

    Retourne
    --------
    mse : float
        Mean Squared Error.
    r2 : float
        R².
    coefficients : list or None
        Les coefficients si le modèle est linéaire (None sinon).
    df_results : pd.DataFrame
        DataFrame enrichi avec les prédictions.
    """
    # Récupérer les données depuis la base
    results = session.query(
        cook.Recipe.recipe_id,
        cook.Recipe.minutes,
        cook.Recipe.n_steps,
        cook.Recipe.n_ingredients,
        func.char_length(cook.Recipe.steps).label("len_steps"),
        func.char_length(cook.Recipe.description).label("len_description"),
    ).all()

    # Conversion des résultats en DataFrame
    df = pd.DataFrame(
        results,
        columns=[
            "recipe_id",
            "minutes",
            "n_steps",
            "n_ingredients",
            "len_steps",
            "len_description",
        ],
    )
    df_results = delete_outliers(df, "minutes", selected_method)

    # Variables indépendantes et cible
    features = ["n_steps", "n_ingredients", "len_steps"]
    target = "minutes"

    X = df_results[features]
    y = df_results[target]

    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Choix du modèle
    coefficients = None
    if model == "rl":
        # Modèle de régression linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    elif model == "xgb":
        # Modèle de gradient boosting
        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    elif model == "rf":
        # Modèle de forêt aléatoire
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        raise ValueError("Modèle non reconnu : choisissez 'rl', 'xgb' ou 'rf'.")

    # Ajout des prédictions au DataFrame
    df_results["predicted_minutes"] = model.predict(X)

    return mse, r2, coefficients, df_results


def delete_outliers(df, key, method, **kwargs):
    """
    Supprime différentes entrées de la clé en fonction de la méthode.

    Paramètres
    ----------
    df : pd.DataFrame
        Jeu de données.
    key : str
        Paramètre sur lequel faire la réduction.
    method : str
        Méthode à utiliser ("DeleteQ1Q3", "Capping", "Log", "Isolation Forest", "DBScan", "Local Outlier Factor").

    Retourne
    --------
    pd.DataFrame
        Le dataset réduit.
    """

    if method == "DeleteQ1Q3":
        Q1 = df[key].quantile(0.25)
        Q3 = df[key].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[key] >= lower_bound) & (df[key] <= upper_bound)]
    elif method == "Capping":
        Q1 = df[key].quantile(0.25)
        Q3 = df[key].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[key] = df[key].clip(lower=lower_bound, upper=upper_bound)
    elif method == "Log":
        df[key] = np.log1p(df[key])
    elif method == "Isolation Forest":
        iso = IsolationForest(
            contamination=kwargs.get("contamination", 0.01), random_state=42
        )
        outliers = iso.fit_predict(df[["minutes", "n_steps", "n_ingredients"]])
        df["is_outlier"] = outliers == -1
        df = df[~df["is_outlier"]].drop(columns=["is_outlier"])
    elif method == "DBScan":
        db = DBSCAN(eps=kwargs.get("eps", 3), min_samples=kwargs.get("min_samples", 5))
        labels = db.fit_predict(df[["minutes", "n_steps"]])
        df["is_outlier"] = labels == -1
        df = df[~df["is_outlier"]].drop(columns=["is_outlier"])
    elif method == "Local Outlier Factor":
        lof = LocalOutlierFactor(
            n_neighbors=kwargs.get("n_neighbors", 20),
            contamination=kwargs.get("contamination", 0.01),
        )
        outliers = lof.fit_predict(df[["minutes", "n_steps"]])
        df["is_outlier"] = outliers == -1
        df = df[~df["is_outlier"]].drop(columns=["is_outlier"])
    else:
        raise ValueError(f"Unknown method: {method}")
    return df
