"""
Module for managing the cooking database models and sessions.

This module defines the database models and provides functions to interact with the database.
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

# Charger les variables d'environnement
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# Connexion à la base de données PostgreSQL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Configuration de la session SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()


# Meta classe pour avoir des getters lisibles des objets
class ObjectCollection:
    """
    Pour convertir la collection d'objets en DataFrame et pour itérer.
    """

    def __init__(self, objects):
        """
        Initialize the ObjectCollection.

        Parameters
        ----------
        objects : list
            A list of objects to be converted to a DataFrame.
        """
        self.objects = objects

    def to_dataframe(self):
        """
        Convertit la collection d'objets en DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data from the objects.
        """
        data = [obj.as_dict() for obj in self.objects]
        return pd.DataFrame(data)

    def __iter__(self):
        """
        Permet l'itération sur la collection d'objets.

        Returns
        -------
        iterator
            An iterator over the objects in the collection.
        """
        return iter(self.objects)

    def __len__(self):
        """
        Permet d'utiliser len() pour obtenir le nombre d'objets dans la collection.

        Returns
        -------
        int
            The number of objects in the collection.
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

        Parameters
        ----------
        session : Session
            The database session used to query the model.
        filters : dict
            Optional filters to apply to the query.

        Returns
        -------
        ObjectCollection
            A collection of objects that match the query.
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

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data from the object.
        """
        return pd.DataFrame([self.as_dict()])

    def as_dict(self):
        """
        Retourne les attributs de l'objet sous forme de dictionnaire.

        Returns
        -------
        dict
            A dictionary containing the object's attributes.
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
        Charge avec les attributs de la table Contributor et les recettes associées (objet recipe).
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
        Initialize a new instance of Recipe.

        Parameters
        ----------
        session : Session, optional
            The database session used to query the recipe.
        id : int, optional
            The ID of the recipe.
        kwargs : dict
            Additional attributes to set on the recipe.
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

    # Relation avec Recipe via la table recipe_ingredient (table déjà existante)
    recipes = relationship(
        "Recipe", secondary=recipe_ingredient, back_populates="ingredients"
    )

    def __init__(self, session=None, id=None, **kwargs):
        """
        Initialize a new instance of Ingredient.

        Parameters
        ----------
        session : Session, optional
            The database session used to query the ingredient.
        id : int, optional
            The ID of the ingredient.
        kwargs : dict
            Additional attributes to set on the ingredient.
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
        Initialize a new instance of Review.

        Parameters
        ----------
        session : Session, optional
            The database session used to query the review.
        id : int, optional
            The ID of the review.
        kwargs : dict
            Additional attributes to set on the review.
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
        Initialize a new instance of Reviewer.

        Parameters
        ----------
        session : Session, optional
            The database session used to query the reviewer.
        id : int, optional
            The ID of the reviewer.
        kwargs : dict
            Additional attributes to set on the reviewer.
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
