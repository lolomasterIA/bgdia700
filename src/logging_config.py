"""
Module de configuration du logging pour l'application.

Ce module configure et retourne un logger pour l'application, en créant
un répertoire `logs` s'il n'existe pas, et en configurant deux handlers
de logging : un pour les messages de debug et un pour les messages d'erreur.
Les logs de debug sont enregistrés dans `logs/debug.log` et les logs d'erreur
dans `logs/error.log`.
"""

import logging
import os


def setup_logging():
    """
    Configure et retourne un logger pour l'application.

    Cette fonction crée un répertoire `logs` s'il n'existe pas, puis configure
    deux handlers de logging : un pour les messages de debug et un pour les messages
    d'erreur. Les logs de debug sont enregistrés dans `logs/debug.log` et les logs
    d'erreur dans `logs/error.log`.

    Retourne
    --------
    logging.Logger
        Le logger configuré.
    """
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        filename=os.path.join("logs", "debug.log"),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create a logger
    logger = logging.getLogger("user_actions")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to prevent duplication
    logger.handlers.clear()

    # Create a handler for the debug log file
    debug_handler = logging.FileHandler("logs/debug.log")
    debug_handler.setLevel(logging.DEBUG)

    # Create a handler for the error log file
    error_handler = logging.FileHandler("logs/error.log")
    error_handler.setLevel(logging.ERROR)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    debug_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)

    return logger
