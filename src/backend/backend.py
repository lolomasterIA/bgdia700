import src.backend.datalayer.cooking as cook
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

# Charger les variables d'environnement
load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')

# Connexion à la base de données PostgreSQL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Configuration de la session SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()

recette = cook.Recipe(session, id=40893)
print(recette.to_dataframe())

reviews = cook.Review.get_all(session, rating=1)
print(reviews.to_dataframe())

session.close()
