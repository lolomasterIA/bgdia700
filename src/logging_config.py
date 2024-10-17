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

    Returns
    -------
    logging.Logger
        Le logger configuré.

    """
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
