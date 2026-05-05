"""
Service Layer — Logique métier pure.
Orchestre le scoring (depuis DB), les 5 modèles ML (Spark),
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

# Schéma Spark — Toutes les features nécessaires
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
    StructField("acteur_taux_succes", FloatType(), False),
    StructField("acteur_delai_moyen", FloatType(), False),
    StructField("tribunal_delai_moyen", FloatType(), False),
    StructField("procedure_taux_succes", FloatType(), False),
])

STATUT_LABELS = {0: "Recouvré", 1: "En cours", 2: "Échec"}
ACTION_LABELS = {0: "Relance amiable", 1: "Mise en demeure", 2: "Action judiciaire", 3: "Négociation"}
PROCEDURE_LABELS = {0: "Amiable", 1: "Judiciaire"}  # Amiable est la classe majoritaire


class PredictionService:

    @staticmethod
    def predict(request: DossierRequest, db: Session) -> PredictionResponse:
        """
        Pipeline complet automatisé :
        1. Prédiction du Type de Procédure (Modèle 1)
        2. Calcul du scoring avocat/huissier (depuis DB) selon la procédure
        3. Construction du DataFrame Spark
        4. Inférence des 4 autres modèles ML
        """
        spark = model_manager.spark

        # -- Calcul des champs automatisés (Zero-Touch) --
        # Le client n'envoie plus le score de risque, on l'estime via une formule simple pour l'instant
        # Plus tard, un 6ème modèle ML pourrait s'en charger.
        computed_score_risque = min(100, max(1, (request.historique_incidents * 10) + int(request.montant_impaye / 1000)))
        
        # Initialisation à 0 car c'est un nouveau dossier
        computed_nombre_evenements = 0
        computed_nombre_retards = 0
        computed_derniere_action_age_jours = 0
        huissier_id = "HUI-AUTO"

        # -- 1. Prédiction du Type de Procédure --
        row_base = Row(
            client_segment=request.client_segment,
            revenu_estime=float(request.revenu_estime),
            score_risque=computed_score_risque,
            montant_impaye=float(request.montant_impaye),
            type_procedure="Amiable",  # Dummy temporaire
            date_ouverture=request.date_ouverture,
            date_mise_a_jour=request.date_mise_a_jour,
            historique_incidents=request.historique_incidents,
            anciennete_impaye_jours=request.anciennete_impaye_jours,
            nombre_echeances_impayees=request.nombre_echeances_impayees,
            nombre_evenements=computed_nombre_evenements,
            nombre_retards=computed_nombre_retards,
            derniere_action_age_jours=computed_derniere_action_age_jours,
            acteur_taux_succes=0.0,
            acteur_delai_moyen=0.0,
            tribunal_delai_moyen=0.0,
            procedure_taux_succes=0.0,
        )
        df_base = spark.createDataFrame([row_base], schema=SINGLE_ROW_SCHEMA)
        
        pred_proc = model_manager.procedure_classifier.transform(df_base)
        proc_index = int(pred_proc.select("prediction").first()[0])
        type_procedure_predite = PROCEDURE_LABELS.get(proc_index, "Amiable")
        logger.info(f"Procédure prédite par l'IA : {type_procedure_predite}")

        # Pour l'automatisation, on force des acteurs par défaut
        tribunal_id = "NONE"
        avocat_id = "NONE"
        if type_procedure_predite == "Judiciaire":
            tribunal_id = "TRIB-AUTO"
            avocat_id = "AVO-AUTO"

        # -- 2. Scoring depuis PostgreSQL (agrégé par client_segment) --
        acteur_metrics = ScoringService.compute_segment_metrics(db, request.client_segment)
        tribunal_metrics = ScoringService.compute_tribunal_metrics(db, request.client_segment, type_procedure_predite)
        procedure_metrics = ScoringService.compute_procedure_metrics(db, request.client_segment, type_procedure_predite)

        acteur_taux_succes = acteur_metrics["acteur_taux_succes"]
        acteur_delai_moyen = acteur_metrics["acteur_delai_moyen"]
        tribunal_delai_moyen = tribunal_metrics["tribunal_delai_moyen"]
        procedure_taux_succes = procedure_metrics["procedure_taux_succes"]

        score_avocat = ScoringService.compute_score_avocat(acteur_taux_succes, acteur_delai_moyen)
        score_huissier = ScoringService.compute_score_huissier(db, request.client_segment)

        # -- 3. Construction du DataFrame complet --
        row_full = Row(
            client_segment=request.client_segment,
            revenu_estime=float(request.revenu_estime),
            score_risque=computed_score_risque,
            montant_impaye=float(request.montant_impaye),
            type_procedure=type_procedure_predite, # Utilisation de la prédiction !
            date_ouverture=request.date_ouverture,
            date_mise_a_jour=request.date_mise_a_jour,
            historique_incidents=request.historique_incidents,
            anciennete_impaye_jours=request.anciennete_impaye_jours,
            nombre_echeances_impayees=request.nombre_echeances_impayees,
            nombre_evenements=computed_nombre_evenements,
            nombre_retards=computed_nombre_retards,
            derniere_action_age_jours=computed_derniere_action_age_jours,
            acteur_taux_succes=float(acteur_taux_succes),
            acteur_delai_moyen=float(acteur_delai_moyen),
            tribunal_delai_moyen=float(tribunal_delai_moyen),
            procedure_taux_succes=float(procedure_taux_succes),
        )
        df_full = spark.createDataFrame([row_full], schema=SINGLE_ROW_SCHEMA)

        # Feature Engineering
        df_full = df_full.withColumn("delai_final_jours", datediff(col("date_mise_a_jour"), col("date_ouverture")))
        df_full = df_full.withColumn("score_avocat", expr("(acteur_taux_succes * 100) - (delai_final_jours * 0.05)"))
        df_full = df_full.withColumn("statut_final", lit("Recouvré"))  # Dummy
        df_full = df_full.withColumn("next_best_action", lit("Relance amiable"))  # Dummy

        # -- 4. Inférence des 4 autres modèles --
        pred_class = model_manager.classifier.transform(df_full)
        class_row = pred_class.select("prediction", "probability").first()
        statut_label = STATUT_LABELS.get(int(class_row["prediction"]), "Inconnu")
        prob_recouvrement = float(class_row["probability"][0]) if len(class_row["probability"]) > 0 else 0.0

        pred_reg = model_manager.regressor.transform(df_full)
        delai_predit = float(pred_reg.select("prediction").first()[0])

        pred_action = model_manager.next_action.transform(df_full)
        next_action = ACTION_LABELS.get(int(pred_action.select("prediction").first()[0]), "Relance amiable")

        pred_cluster = model_manager.cluster.transform(df_full)
        cluster_id = int(pred_cluster.select("cluster_id").first()[0])

        # -- 5. Persistance PostgreSQL --
        feature_id = str(uuid.uuid4())
        history_record = PredictionHistory(
            feature_id=feature_id,
            dossier_id=request.dossier_id,
            procedure_id=request.procedure_id,
            client_segment=request.client_segment,
            revenu_estime=request.revenu_estime,
            score_risque=computed_score_risque,
            historique_incidents=request.historique_incidents,
            montant_impaye=request.montant_impaye,
            anciennete_impaye_jours=request.anciennete_impaye_jours,
            nombre_echeances_impayees=request.nombre_echeances_impayees,
            type_procedure=type_procedure_predite,
            tribunal_id=tribunal_id,
            avocat_id=avocat_id,
            huissier_id=huissier_id,
            nombre_evenements=computed_nombre_evenements,
            nombre_retards=computed_nombre_retards,
            derniere_action_age_jours=computed_derniere_action_age_jours,
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

        # -- 6. Réponse --
        return PredictionResponse(
            avocat_id=avocat_id,
            huissier_id=huissier_id,
            tribunal_id=tribunal_id,
            meilleure_procedure=type_procedure_predite,
            taux_de_succes=round(prob_recouvrement, 4),
            statut_final_predit=statut_label,
            delai_estime_jours=round(delai_predit, 2),
            prochaine_action_recommandee=next_action,
            cluster_segment_id=cluster_id,
            score_avocat=score_avocat,
            score_huissier=score_huissier,
            acteur_taux_succes=acteur_taux_succes,
            acteur_delai_moyen=acteur_delai_moyen,
        )
