"""
Module for altering the database schema and updating ingredient name with lem_name.
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

# récupère les ingrédient avec les ratings sommés et le nombre de reviews (associé recettes)
query = (
    session.query(
        cook.recipe_ingredient.c.ingredient_id,
        func.sum(cook.Review.rating).label("sum_rating"),
        func.count(cook.Review.review_id).label("count_review"),
    )
    .join(
        cook.reviewer_recipe_review,
        cook.recipe_ingredient.c.recipe_id == cook.reviewer_recipe_review.c.recipe_id,
    )
    .join(cook.Review, cook.reviewer_recipe_review.c.review_id == cook.Review.review_id)
    .group_by(cook.recipe_ingredient.c.ingredient_id)
)

results = query.all()


with engine.begin() as connection:
    connection.execute(
        text(
            "ALTER TABLE ingredient ADD COLUMN IF NOT EXISTS sum_rating INTEGER DEFAULT 0;"
        )
    )
    connection.execute(
        text(
            "ALTER TABLE ingredient ADD COLUMN IF NOT EXISTS count_review INTEGER DEFAULT 0;"
        )
    )

# Mettre à jour les colonnes nb_rating et avg_rating dans la table recipe
for result in results:
    session.execute(
        update(cook.Ingredient)
        .where(cook.Ingredient.ingredient_id == result.ingredient_id)
        .values(sum_rating=result.sum_rating, count_review=result.count_review)
    )

# Valider les modifications
session.commit()

# Fermer la session
session.close()
