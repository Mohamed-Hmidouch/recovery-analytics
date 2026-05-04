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
    Singleton global pour les 5 modèles ML :
      - procedure_classifier : Prédiction type de procédure (Amiable/Judiciaire)
      - classifier  : Probabilité de recouvrement (statut_final)
      - regressor   : Prédiction durée procédure (delai_final_jours)
      - next_action  : Next Best Action
      - cluster     : Segmentation KMeans
    """

    def __init__(self):
        self.spark: SparkSession | None = None
        self.procedure_classifier: PipelineModel | None = None
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
        logger.info("Démarrage du ModelManager (5 modèles)...")
        self.spark = (
            SparkSession.builder
            .appName("SmartRecovery-API-Serving")
            .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        self.procedure_classifier = PipelineModel.load(self._resolve_latest_model("rf_procedure"))
        logger.info("   ✓ Modèle 1/5 chargé : Classification (type_procedure)")

        self.classifier = PipelineModel.load(self._resolve_latest_model("rf_classifier"))
        logger.info("   ✓ Modèle 2/5 chargé : Classification (statut_final)")

        self.regressor = PipelineModel.load(self._resolve_latest_model("rf_regressor"))
        logger.info("   ✓ Modèle 3/5 chargé : Régression (delai_final_jours)")

        self.next_action = PipelineModel.load(self._resolve_latest_model("rf_next_action"))
        logger.info("   ✓ Modèle 4/5 chargé : Next Best Action")

        self.cluster = PipelineModel.load(self._resolve_latest_model("kmeans_cluster"))
        logger.info("   ✓ Modèle 5/5 chargé : Clustering (KMeans)")

        self._loaded = True
        logger.info("ModelManager prêt — 5 modèles en mémoire.")

    def shutdown(self):
        if self.spark is not None:
            self.spark.stop()
        self._loaded = False


model_manager = ModelManager()
