"""
Core — Model Manager (Singleton).
Charge les 4 modèles PySpark ML au démarrage de l'API.
"""

import os
import glob
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class ModelManager:
    """
    Singleton global pour les 4 modèles ML :
      - classifier  : Probabilité de recouvrement (statut_final)
      - regressor   : Prédiction durée procédure (delai_final_jours)
      - next_action  : Next Best Action
      - cluster     : Segmentation KMeans
    """

    def __init__(self):
        self.spark: SparkSession | None = None
        self.classifier: PipelineModel | None = None
        self.regressor: PipelineModel | None = None
        self.next_action: PipelineModel | None = None
        self.cluster: PipelineModel | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _resolve_latest_model(self, prefix: str) -> str:
        models_dir = BASE_DIR / "models"
        pattern = str(models_dir / f"{prefix}_*")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"Aucun modèle trouvé avec le préfixe '{prefix}'.")
        return candidates[-1]

    def startup(self):
        logger.info("Démarrage du ModelManager (4 modèles)...")
        self.spark = (
            SparkSession.builder
            .appName("SmartRecovery-API-Serving")
            .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        self.classifier = PipelineModel.load(self._resolve_latest_model("rf_classifier"))
        logger.info("   ✓ Modèle 1/4 chargé : Classification (statut_final)")

        self.regressor = PipelineModel.load(self._resolve_latest_model("rf_regressor"))
        logger.info("   ✓ Modèle 2/4 chargé : Régression (delai_final_jours)")

        self.next_action = PipelineModel.load(self._resolve_latest_model("rf_next_action"))
        logger.info("   ✓ Modèle 3/4 chargé : Next Best Action")

        self.cluster = PipelineModel.load(self._resolve_latest_model("kmeans_cluster"))
        logger.info("   ✓ Modèle 4/4 chargé : Clustering (KMeans)")

        self._loaded = True
        logger.info("ModelManager prêt — 4 modèles en mémoire.")

    def shutdown(self):
        if self.spark is not None:
            self.spark.stop()
        self._loaded = False


model_manager = ModelManager()
