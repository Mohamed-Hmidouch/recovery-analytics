"""
Core — Model Manager (Singleton).
Charge les modèles PySpark une seule fois au démarrage de l'API (lifespan event).
Équivalent d'un @Bean @Singleton dans Spring Boot.
"""

import os
import glob
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # racine du projet


class ModelManager:
    """
    Gestionnaire Singleton pour la session Spark et les modèles ML.
    Les modèles sont chargés en RAM une seule fois, puis réutilisés pour chaque requête.
    """

    def __init__(self):
        self.spark: SparkSession | None = None
        self.classifier: PipelineModel | None = None
        self.regressor: PipelineModel | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _resolve_latest_model(self, prefix: str) -> str:
        """
        Résout le chemin du modèle le plus récent selon le timestamp dans son nom de dossier.
        Lève FileNotFoundError si aucun modèle n'est trouvé.
        """
        models_dir = BASE_DIR / "models"
        pattern = str(models_dir / f"{prefix}_*")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"Aucun modèle trouvé avec le préfixe '{prefix}' dans {models_dir}. "
                f"Avez-vous exécuté le pipeline d'entraînement ?"
            )
        latest = candidates[-1]
        logger.info(f"Modèle résolu : {latest}")
        return latest

    def startup(self):
        """Initialise Spark et charge les modèles. Appelé au lifespan startup."""
        logger.info("Démarrage du ModelManager...")

        # 1. Session Spark (mode local pour le serving)
        self.spark = SparkSession.builder \
            .appName("SmartRecovery-API-Serving") \
            .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g")) \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        logger.info("Session Spark initialisée pour le serving.")

        # 2. Chargement des modèles les plus récents
        classifier_path = self._resolve_latest_model("rf_classifier")
        regressor_path = self._resolve_latest_model("rf_regressor")

        self.classifier = PipelineModel.load(classifier_path)
        logger.info("RandomForestClassifier chargé avec succès.")

        self.regressor = PipelineModel.load(regressor_path)
        logger.info("RandomForestRegressor chargé avec succès.")

        self._loaded = True
        logger.info("ModelManager prêt — tous les modèles sont en mémoire.")

    def shutdown(self):
        """Libère les ressources. Appelé au lifespan shutdown."""
        if self.spark is not None:
            self.spark.stop()
            logger.info("Session Spark arrêtée proprement.")
        self._loaded = False


# Instance Singleton globale — importée partout dans l'API
model_manager = ModelManager()
