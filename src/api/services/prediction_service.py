"""
Service Layer — Logique métier pure.
Reçoit des données validées (DTO), exécute le Feature Engineering,
appelle les modèles PySpark et retourne un résultat structuré.
Équivalent d'un @Service dans Spring Boot.
"""

import logging
from pyspark.sql import Row
from pyspark.sql.functions import col, datediff, expr, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, DateType
)

from src.api.core.model_manager import model_manager
from src.api.schemas.dossier import DossierRequest, PredictionResponse

logger = logging.getLogger(__name__)

# Schéma Spark explicite pour construire le DataFrame d'une seule ligne
SINGLE_ROW_SCHEMA = StructType([
    StructField("client_segment", StringType(), False),
    StructField("revenu_estime", FloatType(), False),
    StructField("score_risque", IntegerType(), False),
    StructField("montant_impaye", FloatType(), False),
    StructField("type_procedure", StringType(), False),
    StructField("acteur_taux_succes", FloatType(), False),
    StructField("date_ouverture", DateType(), False),
    StructField("date_mise_a_jour", DateType(), False),
])

# Mapping inverse : index -> label du statut final (ordre défini par le StringIndexer du training)
# L'ordre dépend de la fréquence dans les données d'entraînement
STATUT_LABELS = {0: "Recouvré", 1: "En cours", 2: "Échec"}


class PredictionService:
    """
    Service de prédiction. Orchestre la transformation et l'inférence.
    Stateless — toute la logique dépend du ModelManager Singleton.
    """

    @staticmethod
    def predict(request: DossierRequest) -> PredictionResponse:
        """
        Pipeline complet :
        1. Conversion DTO → Spark DataFrame (1 ligne)
        2. Feature Engineering (delai_final_jours, score_avocat)
        3. Inférence Classification (statut_final)
        4. Inférence Régression (delai_final_jours)
        5. Construction de la réponse
        """
        spark = model_manager.spark
        classifier = model_manager.classifier
        regressor = model_manager.regressor

        # -- 1. Conversion DTO → DataFrame Spark d'une seule ligne --
        row = Row(
            client_segment=request.client_segment,
            revenu_estime=float(request.revenu_estime),
            score_risque=int(request.score_risque),
            montant_impaye=float(request.montant_impaye),
            type_procedure=request.type_procedure,
            acteur_taux_succes=float(request.acteur_taux_succes),
            date_ouverture=request.date_ouverture,
            date_mise_a_jour=request.date_mise_a_jour,
        )
        df = spark.createDataFrame([row], schema=SINGLE_ROW_SCHEMA)

        # -- 2. Feature Engineering (identique au train_pipeline.py) --
        df = df.withColumn(
            "delai_final_jours",
            datediff(col("date_mise_a_jour"), col("date_ouverture"))
        )
        df = df.withColumn(
            "score_avocat",
            expr("(acteur_taux_succes * 100) - (delai_final_jours * 0.05)")
        )

        # Ajout d'un statut_final factice (le StringIndexer du pipeline en a besoin en entrée)
        df = df.withColumn("statut_final", lit("Recouvré"))

        # -- 3. Classification --
        pred_class = classifier.transform(df)
        class_row = pred_class.select("prediction", "probability").first()
        predicted_index = int(class_row["prediction"])
        statut_label = STATUT_LABELS.get(predicted_index, "Inconnu")

        # Extraire la probabilité de recouvrement (index 0 = Recouvré)
        prob_vector = class_row["probability"]
        prob_recouvrement = float(prob_vector[0]) if len(prob_vector) > 0 else 0.0

        # -- 4. Régression --
        pred_reg = regressor.transform(df)
        delai_predit = float(pred_reg.select("prediction").first()[0])

        # -- 5. Score Avocat --
        score_av = float(df.select("score_avocat").first()[0])

        logger.info(
            f"Prédiction effectuée — Statut: {statut_label}, "
            f"P(recouvrement): {prob_recouvrement:.3f}, "
            f"Délai estimé: {delai_predit:.1f}j, "
            f"Score avocat: {score_av:.2f}"
        )

        return PredictionResponse(
            statut_final_predit=statut_label,
            probabilite_recouvrement=round(prob_recouvrement, 4),
            delai_estime_jours=round(delai_predit, 2),
            score_avocat=round(score_av, 2),
        )
