"""Le coeur de l'application, s'appuie sur le front et le back."""

from src.logging_config import setup_logging

# Configurer le logger
logger = setup_logging()

# Exemple d'utilisation du logger
logger.debug("This is a debug message")
logger.error("This is an error message")
logger.critical("This is a critical message")

# Votre code principal ici
if __name__ == "__main__":
    logger.info("Application started")
    # ... votre code ...
