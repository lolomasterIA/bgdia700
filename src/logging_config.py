import logging
import os


def setup_logging():
    # Créer le répertoire logs s'il n'existe pas
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Créer un logger
    logger = logging.getLogger("user_actions")
    logger.setLevel(logging.DEBUG)

    # Créer un handler pour le fichier de log de debug
    debug_handler = logging.FileHandler("logs/debug.log")
    debug_handler.setLevel(logging.DEBUG)

    # Créer un handler pour le fichier de log des erreurs
    error_handler = logging.FileHandler("logs/error.log")
    error_handler.setLevel(logging.ERROR)

    # Créer un formatter et l'ajouter aux handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Ajouter les handlers au logger
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)

    return logger
