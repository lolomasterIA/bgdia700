"""
Module for altering the database schema and updating ingredient name with name with only one word in.
"""

from sqlalchemy import create_engine, text, update
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
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


def oneword_ingredient(session):
    """
    Récupère les noms d'ingrédients composés d'un seul mot.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour interroger la base de données.

    Retourne
    -------
    list
        Une liste de noms d'ingrédients composés d'un seul mot.
    """
    query = session.query(cook.Ingredient.name).filter(
        cook.Ingredient.name.op("~")("^[^ ]+$")
    )
    return [name for (name,) in query.all()]


def all_ingredient(session):
    """
    Récupère tous les ingrédients avec leurs identifiants.

    Paramètres
    ----------
    session : Session
        La session SQLAlchemy pour interroger la base de données.

    Retourne
    -------
    list
        Une liste de tuples contenant les noms d'ingrédients et leurs identifiants.
    """
    query = session.query(cook.Ingredient.name, cook.Ingredient.ingredient_id)
    return query.all()


# récupérer tous les ingrédients et tous les ingrédients à un mot.
onewordingr = set(oneword_ingredient(session))
allwordsingr = all_ingredient(session)
newwords = {}
for name, id in allwordsingr:
    for word in name.split():
        if word in onewordingr:
            newwords[id] = word
print("youhou")

# Ajouter une colonne à la table si elle n'existe pas déjà
with engine.begin() as connection:
    connection.execute(
        text("ALTER TABLE ingredient ADD COLUMN IF NOT EXISTS name_one_word VARCHAR;")
    )

# Mettre à jour les valeurs de la colonne name_one_word (plus lemmatize)
for id, one_word in newwords.items():
    session.execute(
        text(
            """
        UPDATE ingredient
        SET name_one_word = :one_word
        WHERE ingredient_id = :id
        """
        ),
        {"one_word": lemmatize_name(one_word), "id": id},
    )

print("ok")
# Valider les changements
session.commit()

# Fermer la session
session.close()
