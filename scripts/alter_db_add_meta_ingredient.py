"""
Module for altering the database schema and updating ingredient name with lem_name.
"""

from sqlalchemy import create_engine, text, update
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import func
import os
from dotenv import load_dotenv
import src.backend.datalayer.cooking as cook
import spacy

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

results = session.query(cook.Ingredient.name,
                        cook.Ingredient.ingredient_id).all()


# Charger un modèle NLP anglais
nlp = spacy.load("en_core_web_sm")


def lemmatize_name(name):
    """
    Lemmatize le mot 'name'.

    Lemmatiser le nom donné en le convertissant en minuscules et en supprimant les mots vides.

    Args:
        name (str): Le nom à lemmatiser.

    Returns
    -------
    str
        La version lemmatisée du nom.
    """

    doc = nlp(name.lower())  # Convertir en minuscules
    # Lemmatiser les tokens
    lemmatized = " ".join(token.lemma_ for token in doc if not token.is_stop)
    return lemmatized


# Traiter les résultats
lem_results = [
    {"ingredient_id": ingredient_id, "lem_name": lemmatize_name(name)}
    for name, ingredient_id in results
]
# Extraire les lem_name uniques
unique_lem_names = {result["lem_name"] for result in lem_results}

# Afficher le nombre de lem_name uniques
print(f"Nombre de lem_name uniques : {len(unique_lem_names)}")
