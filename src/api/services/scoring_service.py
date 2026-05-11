"""
Service de Scoring — Calcule les métriques de performance depuis PostgreSQL.
Toutes les métriques sont agrégées par client_segment (profil de risque réel).
Les constantes DEFAULT_* servent uniquement de fallback cold-start (historique vide).
Équivalent d'un @Service spécialisé en Spring Boot.
"""

import logging
from sqlalchemy.orm import Session

from src.api.db.models import PredictionHistory

logger = logging.getLogger(__name__)

# --- Fallback Cold-Start uniquement (roue de secours quand l'historique est vide) ---
DEFAULT_TAUX_SUCCES = 0.65
DEFAULT_DELAI_MOYEN = 120.0
DEFAULT_TRIBUNAL_DELAI = 180.0
DEFAULT_PROCEDURE_TAUX = 0.75


class ScoringService:
    """
    Calcule le scoring depuis l'historique réel de prediction_history.
    La logique centrale est agrégée par client_segment :
    le score reflète ce qui s'est réellement passé pour ce profil de client.
    """

    @staticmethod
    def compute_segment_metrics(db: Session, client_segment: str) -> dict:
        """
        Calcule le taux de succès et le délai moyen pour un segment client
        en agrégeant TOUS les dossiers historiques du même segment.
        Priorité : données réelles → fallback cold-start si historique vide.
        """
        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.client_segment == client_segment)
            .all()
        )

        if not records:
            logger.info(
                f"Cold-start : aucun historique pour le segment '{client_segment}'. "
                f"Utilisation des valeurs de démarrage."
            )
            return {
                "acteur_taux_succes": DEFAULT_TAUX_SUCCES,
                "acteur_delai_moyen": DEFAULT_DELAI_MOYEN,
            }

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        delai_moy = sum(r.delai_estime_jours for r in records) / total

        logger.info(
            f"Scoring segment '{client_segment}' : "
            f"{succes}/{total} succès, délai moyen {round(delai_moy, 1)}j"
        )
        return {
            "acteur_taux_succes": round(succes / total, 3),
            "acteur_delai_moyen": round(delai_moy, 1),
        }

    @staticmethod
    def compute_tribunal_metrics(db: Session, client_segment: str, type_procedure: str) -> dict:
        """
        Calcule le délai moyen au tribunal pour un segment donné.
        N'a de sens que pour les procédures Judiciaires.
        """
        if type_procedure != "Judiciaire":
            return {"tribunal_delai_moyen": 0.0}

        records = (
            db.query(PredictionHistory)
            .filter(
                PredictionHistory.client_segment == client_segment,
                PredictionHistory.type_procedure == "Judiciaire",
            )
            .all()
        )

        if not records:
            logger.info(
                f"Cold-start tribunal : aucun historique Judiciaire pour segment '{client_segment}'."
            )
            return {"tribunal_delai_moyen": DEFAULT_TRIBUNAL_DELAI}

        delai_moy = sum(r.delai_estime_jours for r in records) / len(records)
        return {"tribunal_delai_moyen": round(delai_moy, 1)}

    @staticmethod
    def compute_procedure_metrics(db: Session, client_segment: str, type_procedure: str) -> dict:
        """
        Calcule le taux de succès pour un type de procédure au sein d'un segment.
        """
        records = (
            db.query(PredictionHistory)
            .filter(
                PredictionHistory.client_segment == client_segment,
                PredictionHistory.type_procedure == type_procedure,
            )
            .all()
        )

        if not records:
            logger.info(
                f"Cold-start procédure : aucun historique '{type_procedure}' "
                f"pour segment '{client_segment}'."
            )
            return {"procedure_taux_succes": DEFAULT_PROCEDURE_TAUX}

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        return {"procedure_taux_succes": round(succes / total, 3)}

    @staticmethod
    def compute_score_avocat(taux_succes: float, delai_moyen: float) -> float:
        """Score composite avocat basé sur les métriques du segment (formule métier)."""
        return round((taux_succes * 100) - (delai_moyen * 0.05), 2)

    @staticmethod
    def compute_score_huissier(db: Session, client_segment: str) -> float:
        """
        Score de performance huissier agrégé par segment client.
        Reflète l'efficacité réelle des huissiers sur ce profil de dossier.
        """
        records = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.client_segment == client_segment)
            .all()
        )

        if not records:
            return round(DEFAULT_TAUX_SUCCES * 100, 2)

        total = len(records)
        succes = sum(1 for r in records if r.statut_predit == "Recouvré")
        taux = succes / total
        delai_moy = sum(r.delai_estime_jours for r in records) / total
        return round((taux * 100) - (delai_moy * 0.05), 2)
