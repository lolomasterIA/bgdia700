"""Création dataframes à partir des fichiers de données."""

import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
data_dir = Path(os.getenv("DATA_DIR", "./data"))


class DataLayerException(Exception):
    """Classe de base pour les exceptions de DataLayer."""


class FileUnreadableError(DataLayerException):
    """Exception levée lorsque le fichier n'est pas lisible."""


class DataLayer:
    """objet regroupant les structures (dataframe) des fichiers de données."""

    def __init__(self):
        """Créer les variable pour chaque fichier à importer."""
        self.interactions_test = None
        self.interactions_train = None
        self.interactions_validation = None
        self.pp_recipes = None
        self.pp_users = None
        self.raw_interactions = None
        self.raw_recipes = None
        self.ingr_map = None

    def load_csv(self, file_path):
        """Charge un fichier CSV en DataFrame pandas."""
        try:
            # Vérifie si le fichier existe
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

            # Vérifie si le fichier est lisible
            if not os.access(file_path, os.R_OK):
                raise FileUnreadableError(f"Le fichier {file_path} n'est pas lisible.")

            # Charge le fichier CSV avec pandas
            return pd.read_csv(file_path)

        except pd.errors.EmptyDataError as e:
            # Lève une exception spécifique pour les fichiers vides ou corrompus
            raise FileUnreadableError(
                f"Le fichier {file_path} est vide ou corrompu."
            ) from e

        except FileNotFoundError as e:
            # Laisse la FileNotFoundError remonter sans la masquer
            raise e

        except FileUnreadableError as e:
            # Laisse la FileUnreadableError remonter sans la masquer
            raise e

        except Exception as e:
            # Toutes les autres exceptions sont encapsulées dans DataLayerException
            raise DataLayerException(
                f"Erreur lors du chargement du fichier CSV {file_path}: {str(e)}"
            ) from e

    def load_pickle(self, file_path):
        """Charge un fichier pickle."""
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

            # Vérifier si le fichier est lisible
            if not os.access(file_path, os.R_OK):
                raise FileUnreadableError(f"Le fichier {file_path} n'est pas lisible.")

            # Tente de charger le fichier pickle avec pandas
            with open(file_path, "rb") as file:
                return pd.read_pickle(file)  # nosec B301

        except pd.errors.EmptyDataError as e:
            # Gestion des fichiers pickle vides ou corrompus
            raise FileUnreadableError(
                f"Erreur lors de la lecture du fichier pickle {file_path}."
            ) from e

        except FileNotFoundError as e:
            # Remonte FileNotFoundError si le fichier n'existe pas
            raise e

        except FileUnreadableError as e:
            # Remonte FileUnreadableError si le fichier n'est pas lisible
            raise e

        except Exception as e:
            # Gère toutes les autres erreurs comme une exception générique de DataLayer
            raise DataLayerException(
                f"Erreur lors du chargement du fichier pickle {file_path}: {str(e)}"
            ) from e

    def load_data(self):
        """Charge tous les fichiers de données."""
        # Charger les fichiers CSV
        self.interactions_test = self.load_csv(str(data_dir) + "/interactions_test.csv")
        self.interactions_train = self.load_csv(
            str(data_dir) + "/interactions_train.csv"
        )
        self.interactions_validation = self.load_csv(
            str(data_dir) + "/interactions_validation.csv"
        )
        self.pp_recipes = self.load_csv(str(data_dir) + "/PP_recipes.csv")
        self.pp_users = self.load_csv(str(data_dir) + "/PP_users.csv")
        self.raw_interactions = self.load_csv(str(data_dir) + "/RAW_interactions.csv")
        self.raw_recipes = self.load_csv(str(data_dir) + "/RAW_recipes.csv")

        # Charger le fichier pickle
        self.ingr_map = self.load_pickle(str(data_dir) + "/ingr_map.pkl")

    # Recupere les données des fichiers et les ajoute dans un dataframe
    def get_interactions_test(self):
        """Getter interaction_test."""
        return self.interactions_test

    def get_interactions_train(self):
        """Getter interactions_train."""
        return self.interactions_train

    def get_interactions_validation(self):
        """Récupère les données du fichier interactions_validation."""
        return self.interactions_validation

    def get_pp_recipes(self):
        """Récupère les données du fichier pp_recipes."""
        return self.pp_recipes

    def get_pp_users(self):
        """Récupère les données du fichier pp_users."""
        return self.pp_users

    def get_raw_interactions(self):
        """Récupère les données du fichier raw_interactions."""
        return self.raw_interactions

    def get_raw_recipes(self):
        """Récupère les données du fichier raw_recipes."""
        return self.raw_recipes

    def get_ingr_map(self):
        """Récupère les données du fichier ingr_map."""
        return self.ingr_map
