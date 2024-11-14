"""
Module for generating the cooking database.

This module connects to the PostgreSQL database, creates necessary tables,
and populates them with data from the DataLayer.
"""

import psycopg2
import pandas as pd
from src.backend.datalayer.initdata import DataLayer

# import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import ast

load_dotenv()

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
)
cur = conn.cursor()
# Créer une instance de la classe DataLayer
data_layer = DataLayer()

# Charger toutes les données
data_layer.load_data()

# Création des tables
# Création des tables avec contributor_id et recipe_id comme clés primaires
cur.execute(
    """
DROP TABLE contributor CASCADE;
CREATE TABLE contributor (
    contributor_id INT PRIMARY KEY
);

DROP TABLE recipe CASCADE;
CREATE TABLE recipe (
    recipe_id INT PRIMARY KEY,
    submitted TIMESTAMP,
    name VARCHAR(255),
    minutes INT,
    description TEXT,
    n_steps INT,
    steps TEXT,
    n_ingredients INT
);

DROP TABLE review CASCADE;
CREATE TABLE review (
    review_id SERIAL PRIMARY KEY,
    rating INT,
    review TEXT,
    review_date TIMESTAMP
);

DROP TABLE reviewer CASCADE;
CREATE TABLE reviewer (
    reviewer_id INT PRIMARY KEY
);

DROP TABLE ingredient CASCADE;
CREATE TABLE ingredient (
    ingredient_id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE
);

DROP TABLE recipe_ingredient CASCADE;
CREATE TABLE recipe_ingredient (
    recipe_id INT REFERENCES recipe(recipe_id),
    ingredient_id INT REFERENCES ingredient(ingredient_id),
    PRIMARY KEY (recipe_id, ingredient_id)
);

DROP TABLE contributor_recipe CASCADE;
CREATE TABLE contributor_recipe (
    contributor_id INT REFERENCES contributor(contributor_id),
    recipe_id INT REFERENCES recipe(recipe_id),
    PRIMARY KEY (contributor_id, recipe_id)
);

DROP TABLE reviewer_recipe_review CASCADE;
CREATE TABLE reviewer_recipe_review (
    reviewer_id INT REFERENCES reviewer(reviewer_id),
    recipe_id INT REFERENCES recipe(recipe_id),
    review_id INT REFERENCES review(review_id),
    PRIMARY KEY (reviewer_id, recipe_id, review_id)
);
"""
)
conn.commit()

# Insertion des contributeurs
for _, row in data_layer.get_raw_recipes().iterrows():
    cur.execute(
        """
    INSERT INTO contributor (contributor_id)
    VALUES (%s)
    ON CONFLICT (contributor_id) DO NOTHING;
    """,
        (row["contributor_id"],),
    )
conn.commit()
print("ok contrib")

# Insertion des recettes
for _, row in data_layer.get_raw_recipes().iterrows():
    cur.execute(
        """
    INSERT INTO recipe (recipe_id, submitted, name, minutes, description, n_steps, steps, n_ingredients)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (recipe_id) DO NOTHING;
    """,
        (
            row["id"],
            row["submitted"],
            row["name"],
            row["minutes"],
            row["description"],
            row["n_steps"],
            row["steps"],
            row["n_ingredients"],
        ),
    )
conn.commit()
print("ok recette")

# Association contributeur-recette dans la table contributor_recipe
for _, row in data_layer.get_raw_recipes().iterrows():
    cur.execute(
        """
    INSERT INTO contributor_recipe (contributor_id, recipe_id)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING;
    """,
        (row["contributor_id"], row["id"]),
    )
conn.commit()
print("ok association contributeur-recette")

# Insertion des utilisateurs
for _, row in data_layer.get_raw_interactions().iterrows():
    cur.execute(
        """
    INSERT INTO reviewer (reviewer_id)
    VALUES (%s)
    ON CONFLICT (reviewer_id) DO NOTHING;
    """,
        (row["user_id"],),
    )
conn.commit()
print("ok reviewer")

# Insertion des avis (reviews) dans la table review
for _, row in data_layer.get_raw_interactions().iterrows():
    cur.execute(
        """
    INSERT INTO review (rating, review, review_date)
    VALUES (%s, %s, %s)
    RETURNING review_id;
    """,
        (row["rating"], row["review"], row["date"]),
    )

    # Récupérer le review_id inséré
    review_id = cur.fetchone()[0]

    # Associer reviewer, recipe et review dans la table reviewer_recipe_review
    cur.execute(
        """
    INSERT INTO reviewer_recipe_review (reviewer_id, recipe_id, review_id)
    VALUES (%s, %s, %s)
    ON CONFLICT DO NOTHING;
    """,
        (row["user_id"], row["recipe_id"], review_id),
    )
conn.commit()
print("ok review et association ")

# insertion des ingrédients (uniques)
unique_ingredients = set()

# Charger les recettes et extraire les ingrédients uniques
for _, row in data_layer.get_raw_recipes().iterrows():
    ingrdient_list = ast.literal_eval(row["ingredients"])
    for ingredient in ingrdient_list:
        if ingredient in unique_ingredients:
            pass
        else:
            unique_ingredients.add(ingredient)

# Insérer les ingrédients uniques dans la table 'ingredient'
cur.executemany(
    """
    INSERT INTO ingredient (name)
    VALUES (%s)
    ON CONFLICT (name) DO NOTHING;
""",
    [(ingredient,) for ingredient in unique_ingredients],
)

conn.commit()
print("ok ingrédient")

# Association recettes / ingrédients
cur.execute("SELECT ingredient_id, name FROM ingredient")
ingredient_cache = {row[1]: row[0] for row in cur.fetchall()}  # {name: ingredient_id}

# Préparer les associations entre recettes et ingrédients
recipe_ingredient_values = []

# Charger les recettes et associer les ingrédients aux recettes
for _, row in data_layer.get_raw_recipes().iterrows():
    recipe_id = row["id"]
    ingredients = ast.literal_eval(row["ingredients"])
    for ingredient in ingredients:
        if ingredient in ingredient_cache:  # Si l'ingrédient est bien dans le cache
            ingredient_id = ingredient_cache[ingredient]
            recipe_ingredient_values.append((recipe_id, ingredient_id))

# Insérer les associations dans la table recipe_ingredient
cur.executemany(
    """
    INSERT INTO recipe_ingredient (recipe_id, ingredient_id)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING;
""",
    recipe_ingredient_values,
)

# Committer l'insertion des associations
conn.commit()

# Fermeture de la connexion
cur.close()
conn.close()
