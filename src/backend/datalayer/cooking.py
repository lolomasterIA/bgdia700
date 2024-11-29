"""
Module de gestion des modèles de base de données de cuisine et des sessions.

Ce module définit les modèles de base de données et fournit des fonctions pour interagir avec la base de données.
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    TIMESTAMP,
    ForeignKey,
    Table,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, joinedload
from dotenv import load_dotenv
import os
import pandas as pd


def load_environment():
    """
    Charge les variables d'environnement à partir d'un fichier .env.

    Retourne
    --------
    dict
        Un dictionnaire contenant les variables d'environnement pour la base de données.
    """
    load_dotenv()
    return {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASS": os.getenv("DB_PASS"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_NAME": os.getenv("DB_NAME"),
    }


def create_db_engine(env):
    """
    Crée et retourne un moteur de base de données SQLAlchemy.

    Paramètres
    ----------
    env : dict
        Un dictionnaire contenant les informations de connexion à la base de données.

    Retourne
    --------
    sqlalchemy.engine.Engine
        Un moteur de base de données SQLAlchemy.
    """
    DATABASE_URL = f"postgresql://{env['DB_USER']}:{env['DB_PASS']}@{env['DB_HOST']}:5432/{env['DB_NAME']}"
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    session = Session()
    return engine, session, Base


env = load_environment()
engine, session, Base = create_db_engine(env)


# Meta classe pour avoir des getters lisibles des objets
class ObjectCollection:
    """
    Pour convertir la collection d'objets en DataFrame et pour itérer.
    """

    def __init__(self, objects):
        """
        Initialise la collection d'objets.

        Paramètres
        ----------
        objects : list
            Une liste d'objets à convertir en DataFrame.
        """
        self.objects = objects

    def to_dataframe(self):
        """
        Convertit la collection d'objets en DataFrame.

        Retourne
        --------
        pd.DataFrame
            Un DataFrame contenant les données des objets.
        """
        data = [obj.as_dict() for obj in self.objects]
        return pd.DataFrame(data)

    def __iter__(self):
        """
        Permet l'itération sur la collection d'objets.

        Retourne
        --------
        iterator
            Un itérateur sur les objets de la collection.
        """
        return iter(self.objects)

    def __len__(self):
        """
        Permet d'utiliser len() pour obtenir le nombre d'objets dans la collection.

        Retourne
        --------
        int
            Le nombre d'objets dans la collection.
        """
        return len(self.objects)


class BaseModel:
    """
    Récupère tous les objets ou applique des filtres dynamiques.

    Retourne une ObjectCollection qui permet la conversion en DataFrame.
    """

    @classmethod
    def get_all(cls, session, **filters):
        """
        Récupère tous les objets ou applique des filtres dynamiques.

        Retourne une ObjectCollection.

        Paramètres
        ----------
        session : Session
            La session de base de données utilisée pour interroger le modèle.
        filters : dict
            Des filtres optionnels à appliquer à la requête.

        Retourne
        --------
        ObjectCollection
            Une collection d'objets correspondant à la requête.
        """
        query = session.query(cls)

        # Appliquer des filtres dynamiques si présents
        if filters:
            for attr, value in filters.items():
                query = query.filter(getattr(cls, attr) == value)

        # Retourner une collection d'objets encapsulée
        objects = query.all()
        return ObjectCollection(objects)

    def to_dataframe(self):
        """
        Convertit l'instance actuelle de l'objet en un DataFrame à une seule ligne.

        Retourne
        --------
        pd.DataFrame
            Un DataFrame contenant les données de l'objet.
        """
        return pd.DataFrame([self.as_dict()])

    def as_dict(self):
        """
        Retourne les attributs de l'objet sous forme de dictionnaire.

        Retourne
        --------
        dict
            Un dictionnaire contenant les attributs de l'objet.
        """
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# Définir les tables d'association existantes dans la base de données
reviewer_recipe_review = Table(
    "reviewer_recipe_review",
    Base.metadata,
    Column("reviewer_id", Integer, ForeignKey("reviewer.reviewer_id")),
    Column("recipe_id", Integer, ForeignKey("recipe.recipe_id")),
    Column("review_id", Integer, ForeignKey("review.review_id")),
    PrimaryKeyConstraint("reviewer_id", "recipe_id", "review_id"),  # Clé composite
)

contributor_recipe = Table(
    "contributor_recipe",
    Base.metadata,
    Column("contributor_id", Integer, ForeignKey("contributor.contributor_id")),
    Column("recipe_id", Integer, ForeignKey("recipe.recipe_id")),
    PrimaryKeyConstraint("contributor_id", "recipe_id"),
)

recipe_ingredient = Table(
    "recipe_ingredient",
    Base.metadata,
    Column("recipe_id", Integer, ForeignKey("recipe.recipe_id")),
    Column("ingredient_id", Integer, ForeignKey("ingredient.ingredient_id")),
)


# Modèle Contributor
class Contributor(Base, BaseModel):
    """
    Modèle de table Contributor avec relations pour la base de données.
    """

    __tablename__ = "contributor"
    contributor_id = Column(Integer, primary_key=True)

    # Relation avec Recipe via la table contributor_recipe (table déjà existante dans la base de données)
    recipes = relationship(
        "Recipe", secondary=contributor_recipe, back_populates="contributors"
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialise l'instance avec les attributs de la table Contributor et les recettes associées (objet recipe).

        Paramètres
        ----------
        session : Session, optionnel
            La session de base de données utilisée pour interroger le modèle Contributor.
        id : int, optionnel
            L'identifiant du contributeur à charger.
        **kwargs : dict
            D'autres attributs à définir pour l'instance.

        Si `id` et `session` sont fournis, l'instance est chargée avec les attributs du contributeur et ses recettes associées.
        Sinon, les attributs sont définis à partir des arguments `kwargs`.
        """
        if id and session:
            contributor = (
                session.query(Contributor)
                .options(joinedload(Contributor.recipes))
                .filter_by(contributor_id=id)
                .first()
            )
            if contributor:
                self.contributor_id = contributor.contributor_id
                self.recipes = contributor.recipes
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)


# Modèle Recipe
class Recipe(Base, BaseModel):
    """
    Modèle de table Recipe avec relations pour la base de données.

    En particulier, recipe à les ingrédients (objet ingredient) et les reviews (review) comme attribut.
    """

    __tablename__ = "recipe"

    recipe_id = Column(Integer, primary_key=True)
    submitted = Column(TIMESTAMP)
    name = Column(String(255))
    minutes = Column(Integer)
    description = Column(Text)
    n_steps = Column(Integer)
    steps = Column(Text)
    n_ingredients = Column(Integer)
    nb_rating = Column(Integer)
    avg_rating = Column(Float)

    # Relation avec Contributor via la table contributor_recipe
    contributors = relationship(
        "Contributor", secondary=contributor_recipe, back_populates="recipes"
    )

    # Relation avec Ingredient via la table recipe_ingredient
    ingredients = relationship(
        "Ingredient", secondary=recipe_ingredient, back_populates="recipes"
    )

    # Relation avec Review et Reviewer via la table reviewer_recipe_review
    reviews = relationship(
        "Review",
        secondary=reviewer_recipe_review,
        back_populates="recipes",
        overlaps="reviewers",
    )
    reviewers = relationship(
        "Reviewer",
        secondary=reviewer_recipe_review,
        back_populates="recipes",
        overlaps="reviews",
        viewonly=True,
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialise une nouvelle instance de Recipe.

        Paramètres
        ----------
        session : Session, optionnel
            La session de base de données utilisée pour interroger la recette.
        id : int, optionnel
            L'identifiant de la recette.
        kwargs : dict
            D'autres attributs à définir pour la recette.
        """
        if id and session:
            recipe = (
                session.query(Recipe)
                .options(
                    joinedload(Recipe.ingredients),
                    joinedload(Recipe.reviews),
                    joinedload(Recipe.reviewers),
                )
                .filter_by(recipe_id=id)
                .first()
            )
            if recipe:
                session.add(recipe)
                self.recipe_id = recipe.recipe_id
                self.submitted = recipe.submitted
                self.name = recipe.name
                self.minutes = recipe.minutes
                self.description = recipe.description
                self.n_steps = recipe.n_steps
                self.steps = recipe.steps
                self.n_ingredients = recipe.n_ingredients
                self.contributors = recipe.contributors
                self.ingredients = recipe.ingredients
                self.reviews = recipe.reviews
                self.reviewers = recipe.reviewers
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)


# Modèle Ingredient


class Ingredient(Base, BaseModel):
    """
    Modèle de table Ingredient avec relations pour la base de données.
    """

    __tablename__ = "ingredient"

    ingredient_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True)
    nb_recette = Column(Integer)
    sum_rating = Column(Integer)
    count_review = Column(Integer)

    # Relation avec Recipe via la table recipe_ingredient (table déjà existante)
    recipes = relationship(
        "Recipe", secondary=recipe_ingredient, back_populates="ingredients"
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialise une nouvelle instance de Ingredient.

        Paramètres
        ----------
        session : Session, optionnel
            La session de base de données utilisée pour interroger l'ingrédient.
        id : int, optionnel
            L'identifiant de l'ingrédient.
        kwargs : dict
            D'autres attributs à définir pour l'ingrédient.
        """
        if id and session:
            ingredient = (
                session.query(Ingredient)
                .options(joinedload(Ingredient.recipes))
                .filter_by(ingredient_id=id)
                .first()
            )
            if ingredient:
                self.ingredient_id = ingredient.ingredient_id
                self.name = ingredient.name
                self.recipes = ingredient.recipes
                self.nb_recette = ingredient.nb_recette
                self.sum_rating = ingredient.sum_rating
                self.count_review = ingredient.count_review
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)


# Modèle Review
class Review(Base, BaseModel):
    """
    Modèle de table Review avec relations pour la base de données.
    """

    __tablename__ = "review"

    review_id = Column(Integer, primary_key=True, autoincrement=True)
    rating = Column(Integer)
    review = Column(Text)
    review_date = Column(TIMESTAMP)

    # Relation avec Recipe et Reviewer via la table reviewer_recipe_review (table déjà existante)
    recipes = relationship(
        "Recipe",
        secondary=reviewer_recipe_review,
        back_populates="reviews",
        overlaps="reviewers",
    )
    reviewers = relationship(
        "Reviewer",
        secondary=reviewer_recipe_review,
        back_populates="reviews",
        overlaps="recipes",
        viewonly=True,
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialise une nouvelle instance de Review.

        Paramètres
        ----------
        session : Session, optionnel
            La session de base de données utilisée pour interroger la critique.
        id : int, optionnel
            L'identifiant de la critique.
        kwargs : dict
            D'autres attributs à définir pour la critique.
        """
        if id and session:
            review = session.query(Review).filter_by(review_id=id).first()
            if review:
                self.review_id = review.review_id
                self.rating = review.rating
                self.review = review.review
                self.review_date = review.review_date
                self.recipes = review.recipes
                self.reviewers = review.reviewers
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)


# Modèle Reviewer
class Reviewer(Base, BaseModel):
    """
    Modèle de table Reviewer avec relations pour la base de données.
    """

    __tablename__ = "reviewer"

    reviewer_id = Column(Integer, primary_key=True)

    # Relation avec Review et Recipe via la table reviewer_recipe_review (table déjà existante)
    reviews = relationship(
        "Review",
        secondary=reviewer_recipe_review,
        back_populates="reviewers",
        overlaps="recipes",
        viewonly=True,
    )
    recipes = relationship(
        "Recipe",
        secondary=reviewer_recipe_review,
        back_populates="reviewers",
        overlaps="reviews",
        viewonly=True,
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialise une nouvelle instance de Reviewer.

        Paramètres
        ----------
        session : Session, optionnel
            La session de base de données utilisée pour interroger le critique.
        id : int, optionnel
            L'identifiant du critique.
        kwargs : dict
            D'autres attributs à définir pour le critique.
        """
        if id and session:
            reviewer = (
                session.query(Reviewer)
                .options(joinedload(Reviewer.recipes))
                .filter_by(reviewer_id=id)
                .first()
            )
            if reviewer:
                self.reviewer_id = reviewer.reviewer_id
                self.reviews = reviewer.reviews
                self.recipes = reviewer.recipes
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)


session.close()
