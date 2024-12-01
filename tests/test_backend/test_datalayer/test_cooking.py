import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from collections import namedtuple

from src.backend.datalayer.cooking import (
    load_environment,
    create_db_engine,
    Contributor,
    Recipe,
    Ingredient,
    Review,
    Reviewer,
    BaseModel,
    ObjectCollection,
    Base,
)


@pytest.fixture
def mock_env():
    return {
        "DB_USER": "test_user",
        "DB_PASS": "test_pass",
        "DB_HOST": "localhost",
        "DB_NAME": "test_db",
    }


@pytest.fixture
def mock_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_session(mock_engine):
    Session = sessionmaker(bind=mock_engine)
    session = Session()
    yield session
    session.close()


def test_load_environment(mock_env):
    with patch.dict("os.environ", mock_env):
        env = load_environment()
        assert env["DB_USER"] == "test_user"
        assert env["DB_PASS"] == "test_pass"
        assert env["DB_HOST"] == "localhost"
        assert env["DB_NAME"] == "test_db"


def test_create_db_engine(mock_env):
    with patch("src.backend.datalayer.cooking.load_environment", return_value=mock_env):
        engine, session, Base = create_db_engine(mock_env)
        assert engine is not None
        assert session is not None
        assert Base is not None


def test_get_all(mock_session):
    # Simuler les objets retournés par la requête
    MockObject = namedtuple("MockObject", ["id", "name", "value"])
    mock_objects = [
        MockObject(id=1, name="Object1", value=10),
        MockObject(id=2, name="Object2", value=20),
        MockObject(id=3, name="Object3", value=30),
    ]

    mock_query = MagicMock()
    mock_query.all = MagicMock(return_value=mock_objects)
    mock_session.query = MagicMock(return_value=mock_query)

    # Appeler la méthode get_all
    result = BaseModel.get_all(mock_session)
    assert isinstance(result, ObjectCollection)

    # Vérifier que tous les objets sont récupérés
    objects = list(result)
    assert len(objects) == 3
    assert objects[0].name == "Object1"

    # Vérifier que query() a bien été appelé
    # Vérifie que query(BaseModel) a été appelé
    mock_session.query.assert_called_once_with(BaseModel)
    mock_query.all.assert_called_once()  # Vérifie que .all() a été invoqué


def test_get_filtered_objects(mock_session):
    # Ajouter un objet Recipe à la base
    recipe = Recipe(recipe_id=3, name="Filtered Recipe")
    mock_session.add(recipe)
    mock_session.commit()

    # Tester la méthode avec un filtre
    result = mock_session.query(Recipe).filter_by(name="Filtered Recipe").first()
    assert result is not None
    assert result.recipe_id == 3
    assert result.name == "Filtered Recipe"


def test_to_dataframe_single_object(mock_session):
    # Ajouter un objet Recipe à la base
    recipe = Recipe(recipe_id=1, name="Test Recipe")
    mock_session.add(recipe)
    mock_session.commit()

    # Récupérer l'objet et tester to_dataframe
    result = mock_session.query(Recipe).filter_by(recipe_id=1).first()
    assert result is not None  # Vérifier que l'objet est récupéré

    # Convertir en DataFrame
    df = result.to_dataframe()

    # Vérifier le contenu du DataFrame
    assert len(df) == 1  # Une seule ligne
    assert df.iloc[0]["recipe_id"] == 1
    assert df.iloc[0]["name"] == "Test Recipe"


def test_as_dict(mock_session):
    # Ajouter un objet Recipe à la base
    recipe = Recipe(recipe_id=1, name="Test Recipe")
    mock_session.add(recipe)
    mock_session.commit()

    # Récupérer l'objet et tester as_dict
    result = mock_session.query(Recipe).filter_by(recipe_id=1).first()
    assert result is not None  # Vérifier que l'objet est récupéré

    # Convertir en dictionnaire
    obj_dict = result.as_dict()

    # Vérifier le contenu du dictionnaire
    assert isinstance(obj_dict, dict)
    assert obj_dict["recipe_id"] == 1
    assert obj_dict["name"] == "Test Recipe"


def test_contributor_model(mock_session):
    contributor = Contributor(contributor_id=1)
    mock_session.add(contributor)
    mock_session.commit()
    result = mock_session.query(Contributor).filter_by(contributor_id=1).first()
    assert result.contributor_id == 1


def test_recipe_model(mock_session):
    recipe = Recipe(recipe_id=1, name="Test Recipe")
    mock_session.add(recipe)
    mock_session.commit()
    result = mock_session.query(Recipe).filter_by(recipe_id=1).first()
    assert result.recipe_id == 1
    assert result.name == "Test Recipe"


def test_ingredient_model(mock_session):
    ingredient = Ingredient(ingredient_id=1, name="Test Ingredient")
    mock_session.add(ingredient)
    mock_session.commit()
    result = mock_session.query(Ingredient).filter_by(ingredient_id=1).first()
    assert result.ingredient_id == 1
    assert result.name == "Test Ingredient"


def test_review_model(mock_session):
    review = Review(review_id=1, rating=5, review="Great!")
    mock_session.add(review)
    mock_session.commit()
    result = mock_session.query(Review).filter_by(review_id=1).first()
    assert result.review_id == 1
    assert result.rating == 5
    assert result.review == "Great!"


def test_reviewer_model(mock_session):
    reviewer = Reviewer(reviewer_id=1)
    mock_session.add(reviewer)
    mock_session.commit()
    result = mock_session.query(Reviewer).filter_by(reviewer_id=1).first()
    assert result.reviewer_id == 1


def test_object_collection():
    objects = [
        MagicMock(as_dict=lambda: {"id": 1}),
        MagicMock(as_dict=lambda: {"id": 2}),
    ]
    collection = ObjectCollection(objects)
    df = collection.to_dataframe()
    assert len(df) == 2
    assert df.iloc[0]["id"] == 1
    assert df.iloc[1]["id"] == 2
    assert len(collection) == 2
    assert list(iter(collection)) == objects
