import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
data_dir = Path(os.getenv('DATA_DIR', './data'))


class DataLayer:
    def __init__(self):
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
        return pd.read_csv(file_path)

    def load_pickle(self, file_path):
        """Charge un fichier pickle."""
        with open(file_path, 'rb') as file:
            return pd.read_pickle(file)

    def load_data(self):
        """Charge tous les fichiers de données dans les attributs appropriés"""
        # Charger les fichiers CSV
        self.interactions_test = self.load_csv(
            str(data_dir) + "/interactions_test.csv")
        self.interactions_train = self.load_csv(
            str(data_dir) + '/interactions_train.csv')
        self.interactions_validation = self.load_csv(
            str(data_dir) + '/interactions_validation.csv')
        self.pp_recipes = self.load_csv(str(data_dir) + '/PP_recipes.csv')
        self.pp_users = self.load_csv(str(data_dir) + '/PP_users.csv')
        self.raw_interactions = self.load_csv(
            str(data_dir) + '/RAW_interactions.csv')
        self.raw_recipes = self.load_csv(str(data_dir) + '/RAW_recipes.csv')

        # Charger le fichier pickle
        self.ingr_map = self.load_pickle(str(data_dir) + '/ingr_map.pkl')

    # Recupere les données des fichiers et les ajoute dans un dataframe
    def get_interactions_test(self):
        return self.interactions_test

    def get_interactions_train(self):
        return self.interactions_train

    def get_interactions_validation(self):
        return self.interactions_validation

    def get_pp_recipes(self):
        return self.pp_recipes

    def get_pp_users(self):
        return self.pp_users

    def get_raw_interactions(self):
        return self.raw_interactions

    def get_raw_recipes(self):
        return self.raw_recipes

    def get_ingr_map(self):
        return self.ingr_map
