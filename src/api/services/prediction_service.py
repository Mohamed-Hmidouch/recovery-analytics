"""
Service Layer — Logique métier pure.
Orchestre le scoring (depuis DB), les 4 modèles ML (Spark),
et la persistance dans PostgreSQL.
"""

import logging
import uuid
from pyspark.sql import Row
from pyspark.sql.functions import col, datediff, expr, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, DateType,
)
from sqlalchemy.orm import Session

from src.api.core.model_manager import model_manager
from src.api.schemas.dossier import DossierRequest, PredictionResponse
from src.api.services.scoring_service import ScoringService
from src.api.db.models import PredictionHistory

logger = logging.getLogger(__name__)

# Schéma Spark — Toutes les features nécessaires aux modèles
SINGLE_ROW_SCHEMA = StructType([
    StructField("client_segment", StringType(), False),
    StructField("revenu_estime", FloatType(), False),
    StructField("score_risque", IntegerType(), False),
    StructField("montant_impaye", FloatType(), False),
    StructField("type_procedure", StringType(), False),
    StructField("date_ouverture", DateType(), False),
    StructField("date_mise_a_jour", DateType(), False),
    StructField("historique_incidents", IntegerType(), False),
    StructField("anciennete_impaye_jours", IntegerType(), False),
    StructField("nombre_echeances_impayees", IntegerType(), False),
    StructField("nombre_evenements", IntegerType(), False),
    StructField("nombre_retards", IntegerType(), False),
    StructField("derniere_action_age_jours", IntegerType(), False),
    # Métriques calculées depuis PostgreSQL (injectées par le service)
    StructField("acteur_taux_succes", FloatType(), False),
    StructField("acteur_delai_moyen", FloatType(), False),
    StructField("tribunal_delai_moyen", FloatType(), False),
    StructField("procedure_taux_succes", FloatType(), False),
])

STATUT_LABELS = {0: "Recouvré", 1: "En cours", 2: "Échec"}
ACTION_LABELS = {0: "Relance amiable", 1: "Mise en demeure", 2: "Action judiciaire", 3: "Négociation"}


class PredictionService:

    @staticmethod
    def predict(request: DossierRequest, db: Session) -> PredictionResponse:
        """
        Pipeline complet :
        1. Calcul du scoring avocat/huissier depuis PostgreSQL
        2. Construction du DataFrame Spark avec les métriques injectées
        3. Feature Engineering
        4. Inférence des 4 modèles ML
        5. Persistance dans PostgreSQL
        6. Retour de la réponse complète
        """
        spark = model_manager.spark

        # -- 1. Scoring depuis PostgreSQL --
        acteur_metrics = ScoringService.compute_acteur_metrics(db, request.avocat_id)
        tribunal_metrics = ScoringService.compute_tribunal_metrics(db, request.tribunal_id)
        procedure_metrics = ScoringService.compute_procedure_metrics(db, request.type_procedure)

        acteur_taux_succes = acteur_metrics["acteur_taux_succes"]
        acteur_delai_moyen = acteur_metrics["acteur_delai_moyen"]
        tribunal_delai_moyen = tribunal_metrics["tribunal_delai_moyen"]
        procedure_taux_succes = procedure_metrics["procedure_taux_succes"]

        score_avocat = ScoringService.compute_score_avocat(acteur_taux_succes, acteur_delai_moyen)
        score_huissier = ScoringService.compute_score_huissier(db, request.huissier_id)

        # -- 2. Construction du DataFrame Spark --
        row = Row(
            client_segment=request.client_segment,
            revenu_estime=float(request.revenu_estime),
            score_risque=int(request.score_risque),
            montant_impaye=float(request.montant_impaye),
            type_procedure=request.type_procedure,
            date_ouverture=request.date_ouverture,
            date_mise_a_jour=request.date_mise_a_jour,
            historique_incidents=request.historique_incidents,
            anciennete_impaye_jours=request.anciennete_impaye_jours,
            nombre_echeances_impayees=request.nombre_echeances_impayees,
            nombre_evenements=request.nombre_evenements,
            nombre_retards=request.nombre_retards,
            derniere_action_age_jours=request.derniere_action_age_jours,
            acteur_taux_succes=float(acteur_taux_succes),
            acteur_delai_moyen=float(acteur_delai_moyen),
            tribunal_delai_moyen=float(tribunal_delai_moyen),
            procedure_taux_succes=float(procedure_taux_succes),
        )
        df = spark.createDataFrame([row], schema=SINGLE_ROW_SCHEMA)

        # -- 3. Feature Engineering --
        df = df.withColumn("delai_final_jours", datediff(col("date_mise_a_jour"), col("date_ouverture")))
        df = df.withColumn("score_avocat", expr("(acteur_taux_succes * 100) - (delai_final_jours * 0.05)"))
        df = df.withColumn("statut_final", lit("Recouvré"))  # Dummy pour le StringIndexer
        df = df.withColumn("next_best_action", lit("Relance amiable"))  # Dummy pour le StringIndexer

        # -- 4. Inférence des 4 modèles --
        # Modèle 1 : Classification (statut_final)
        pred_class = model_manager.classifier.transform(df)
        class_row = pred_class.select("prediction", "probability").first()
        predicted_index = int(class_row["prediction"])
        statut_label = STATUT_LABELS.get(predicted_index, "Inconnu")
        prob_vector = class_row["probability"]
        prob_recouvrement = float(prob_vector[0]) if len(prob_vector) > 0 else 0.0

        # Modèle 2 : Régression (delai_final_jours)
        pred_reg = model_manager.regressor.transform(df)
        delai_predit = float(pred_reg.select("prediction").first()[0])

        # Modèle 3 : Next Best Action
        pred_action = model_manager.next_action.transform(df)
        action_index = int(pred_action.select("prediction").first()[0])
        next_action = ACTION_LABELS.get(action_index, "Relance amiable")

        # Modèle 4 : KMeans Clustering
        pred_cluster = model_manager.cluster.transform(df)
        cluster_id = int(pred_cluster.select("cluster_id").first()[0])

        # -- 5. Persistance PostgreSQL --
        feature_id = str(uuid.uuid4())
        history_record = PredictionHistory(
            feature_id=feature_id,
            dossier_id=request.dossier_id,
            procedure_id=request.procedure_id,
            client_segment=request.client_segment,
            revenu_estime=request.revenu_estime,
            score_risque=request.score_risque,
            historique_incidents=request.historique_incidents,
            montant_impaye=request.montant_impaye,
            anciennete_impaye_jours=request.anciennete_impaye_jours,
            nombre_echeances_impayees=request.nombre_echeances_impayees,
            type_procedure=request.type_procedure,
            tribunal_id=request.tribunal_id,
            avocat_id=request.avocat_id,
            huissier_id=request.huissier_id,
            nombre_evenements=request.nombre_evenements,
            nombre_retards=request.nombre_retards,
            derniere_action_age_jours=request.derniere_action_age_jours,
            acteur_taux_succes=acteur_taux_succes,
            acteur_delai_moyen=acteur_delai_moyen,
            tribunal_delai_moyen=tribunal_delai_moyen,
            procedure_taux_succes=procedure_taux_succes,
            cluster_segment_id=cluster_id,
            statut_predit=statut_label,
            probabilite_recouvrement=prob_recouvrement,
            delai_estime_jours=delai_predit,
            score_avocat=score_avocat,
            next_best_action=next_action,
        )

        db.add(history_record)
        db.commit()
        db.refresh(history_record)
        logger.info(f"Prédiction historisée (ID: {history_record.id}, Dossier: {request.dossier_id})")

        # -- 6. Réponse --
        return PredictionResponse(
            statut_final_predit=statut_label,
            probabilite_recouvrement=round(prob_recouvrement, 4),
            delai_estime_jours=round(delai_predit, 2),
            next_best_action=next_action,
            cluster_segment_id=cluster_id,
            score_avocat=score_avocat,
            score_huissier=score_huissier,
            acteur_taux_succes=acteur_taux_succes,
            acteur_delai_moyen=acteur_delai_moyen,
        )
