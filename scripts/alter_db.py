"""
Module for altering the database schema and updating ingredient and recipe statistics.

This module connects to the PostgreSQL database, adds necessary columns to the tables,
and updates the statistics for ingredients and recipes.
"""

from sqlalchemy import create_engine, text, update
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import func
import os
from dotenv import load_dotenv
import src.backend.datalayer.cooking as cook

# Charger les variables d'environnement
load_dotenv()

# Connexion à la base de données PostgreSQL cooking
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

# Ajouter la colonne nb_recette à la table ingredient si elle n'existe pas déjà
with engine.begin() as connection:
    connection.execute(
        text(
            "ALTER TABLE ingredient ADD COLUMN IF NOT EXISTS nb_recette INTEGER DEFAULT 0;"
        )
    )

# Calculer le nombre de recettes pour chaque ingrédient
results = (
    session.query(
        cook.recipe_ingredient.c.ingredient_id,
        func.count(cook.recipe_ingredient.c.recipe_id).label("nb_recette"),
    )
    .group_by(cook.recipe_ingredient.c.ingredient_id)
    .all()
)

# Mettre à jour la colonne nb_recette dans la table ingredient
for result in results:
    session.execute(
        update(cook.Ingredient)
        .where(cook.Ingredient.ingredient_id == result.ingredient_id)
        .values(nb_recette=result.nb_recette)
    )

# Valider les modifications
session.commit()

# Ajouter les colonnes nb_rating et avg_rating à recipe si elles n'existent pas déjà
with engine.begin() as connection:
    connection.execute(
        text("ALTER TABLE recipe ADD COLUMN IF NOT EXISTS nb_rating INTEGER DEFAULT 0;")
    )
    connection.execute(
        text("ALTER TABLE recipe ADD COLUMN IF NOT EXISTS avg_rating FLOAT DEFAULT 0;")
    )

recipe_results = (
    session.query(
        cook.reviewer_recipe_review.c.recipe_id,
        func.count(cook.reviewer_recipe_review.c.review_id).label("nb_rating"),
        func.avg(cook.Review.rating).label("avg_rating"),
    )
    .join(cook.Review, cook.reviewer_recipe_review.c.review_id == cook.Review.review_id)
    .group_by(cook.reviewer_recipe_review.c.recipe_id)
    .all()
)

# Mettre à jour les colonnes nb_rating et avg_rating dans la table recipe
for result in recipe_results:
    session.execute(
        update(cook.Recipe)
        .where(cook.Recipe.recipe_id == result.recipe_id)
        .values(nb_rating=result.nb_rating, avg_rating=result.avg_rating)
    )

# Valider les modifications
session.commit()

# Fermer la session
session.close()
