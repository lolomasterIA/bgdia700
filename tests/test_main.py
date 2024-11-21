import os
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import src.main as main
from src.main import load_environment, create_db_engine


def test_load_environment():
    """Test environment variable loading"""
    env = load_environment()

    # Check that all required keys are present
    required_keys = ["DB_USER", "DB_PASS", "DB_HOST", "DB_NAME"]
    for key in required_keys:
        assert key in env, f"{key} not found in environment variables"
        assert env[key] is not None, f"{key} is None"


def test_create_db_engine():
    """Test database engine creation"""
    # Load environment first
    env = load_environment()

    # Create engine
    engine, session, Base = create_db_engine(env)

    # Assertions
    assert engine is not None, "Engine creation failed"
    assert session is not None, "Session creation failed"

    # Optional: Test connection
    try:
        with engine.connect() as connection:
            assert connection is not None, "Could not establish database connection"
    except Exception as e:
        pytest.fail(f"Database connection error: {e}")
    finally:
        session.close()
