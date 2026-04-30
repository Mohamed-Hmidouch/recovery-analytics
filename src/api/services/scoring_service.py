"""
Service de Scoring — Calcule les métriques de performance depuis PostgreSQL.
Remplace les valeurs hardcodées que le client envoyait manuellement.
Équivalent d'un @Service spécialisé en Spring Boot.
"""

import logging
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.api.db.models import PredictionHistory

logger = logging.getLogger(__name__)

# Valeurs par défaut utilisées quand la DB n'a pas encore d'historique
# pour un acteur donné (démarrage à froid)
DEFAULT_TAUX_SUCCES = 0.65
DEFAULT_DELAI_MOYEN = 120.0
DEFAULT_TRIBUNAL_DELAI = 180.0
DEFAULT_PROCEDURE_TAUX = 0.75


class ScoringService:
    """
    Calcule le scoring avocat/huissier/tribunal
    en agrégeant les données historiques de prediction_history.
    """

    @staticmethod
    def compute_acteur_metrics(db: Session, avocat_id: str) -> dict:
        """
        Calcule le taux de succès et le délai moyen d'un avocat
        depuis l'historique des prédictions.
        """
        if avocat_id == "NONE":
            return {
                "acteur_taux_succes": DEFAULT_TAUX_SUCCES,
                "acteur_delai_moyen": DEFAULT_DELAI_MOYEN,
            }

        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.avocat_id == avocat_id)
            .all()
        )

        if not records:
            logger.info(f"Aucun historique pour avocat '{avocat_id}'. Utilisation des défauts.")
            return {
                "acteur_taux_succes": DEFAULT_TAUX_SUCCES,
                "acteur_delai_moyen": DEFAULT_DELAI_MOYEN,
            }

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        delai_moy = sum(r.delai_estime_jours for r in records) / total

        return {
            "acteur_taux_succes": round(succes / total, 3),
            "acteur_delai_moyen": round(delai_moy, 1),
        }

    @staticmethod
    def compute_tribunal_metrics(db: Session, tribunal_id: str) -> dict:
        """Calcule le délai moyen d'un tribunal."""
        if tribunal_id == "NONE":
            return {"tribunal_delai_moyen": 0.0}

        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.tribunal_id == tribunal_id)
            .all()
        )

        if not records:
            return {"tribunal_delai_moyen": DEFAULT_TRIBUNAL_DELAI}

        delai_moy = sum(r.delai_estime_jours for r in records) / len(records)
        return {"tribunal_delai_moyen": round(delai_moy, 1)}

    @staticmethod
    def compute_procedure_metrics(db: Session, type_procedure: str) -> dict:
        """Calcule le taux de succès pour un type de procédure."""
        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.type_procedure == type_procedure)
            .all()
        )

        if not records:
            return {"procedure_taux_succes": DEFAULT_PROCEDURE_TAUX}

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        return {"procedure_taux_succes": round(succes / total, 3)}

    @staticmethod
    def compute_score_avocat(taux_succes: float, delai_moyen: float) -> float:
        """Score composite de l'avocat (formule métier)."""
        return round((taux_succes * 100) - (delai_moyen * 0.05), 2)

    @staticmethod
    def compute_score_huissier(db: Session, huissier_id: str) -> float:
        """Score de performance de l'huissier."""
        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.huissier_id == huissier_id)
            .all()
        )

        if not records:
            return round(DEFAULT_TAUX_SUCCES * 100, 2)

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        taux = succes / total
        delai_moy = sum(r.delai_estime_jours for r in records) / total
        return round((taux * 100) - (delai_moy * 0.05), 2)
