import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

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
