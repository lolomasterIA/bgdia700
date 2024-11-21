import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from src.backend.datalayer.cooking import Base
from src.main import (
    load_environment,
    create_db_engine,
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
