# bgdia700
Cette application génère une interface streamlit pour explorer les données issues des fichiers :
- data\RAW_interactions.csv
- data\RAW_recipes.csv

Structure du projet :
- docs/ : Documentation du projet.
- scripts/ : Scripts utilitaires pour des tâches diverses (importation de données, nettoyage, etc.). Cela permet en particulier la création de la base de données utilisée dans l'application.
- src/ : Code source principal de l'application.
- src/backend/datalayer/ : couche d'accès au données. En particulier, cooking.py définit l'ensemble des classes qui permettent de manipuler les objets de la base de données.
- src/backend/ : backend.py regroupe l'ensemble des méthodes permettant le calcul et la mise en forme des données.
- src/frontend/ : fonctions et classes nécessaire pour l'affichage de l'application
- tests/ : Tests unitaires et d'intégration pour assurer la qualité du code.

Installation :
Données :
L'application utilise une base de données postgres.
0/ installer un sgbd postgres et renseigner le fichier .env :
- DATA_DIR=./data
- DB_HOST="localhost"
- DB_NAME=""
- DB_USER=""
- DB_PASS=""
1/ Utiliser le dump cooking.sql (à jour) pour créer la base dans PG (cooking)
2/ régénérer la base de données depuis les scripts, pour cela exécuter les scripts dans l'ordre :
- scripts\generate_db_cooking.py
- scripts\alter_db.py
- scripts\alter_db_add_rating_ingredient.py
- scripts\alter_db_add_meta_ingredient.py

Applicatif :
installer poetry
installer les dépendances poetry install

Pour lancer : streamlit run src/main.py
