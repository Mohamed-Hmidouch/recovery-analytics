"""
Schemas Pydantic (DTOs) — Validation stricte des Entrées/Sorties.

Le client envoie UNIQUEMENT les données brutes du dossier.
Les métriques de performance (acteur_taux_succes, etc.) sont calculées par le système.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import date, datetime


class DossierRequest(BaseModel):
    """
    Données d'entrée envoyées par le client.
    NE CONTIENT PAS les métriques de performance (calculées automatiquement par le système).
    """
    # Identifiants
    dossier_id: str = Field(..., description="ID unique du dossier")
    procedure_id: str = Field(..., description="ID unique de la procédure")

    # Client
    client_segment: Literal["Retail", "Professionnel", "Corporate"] = Field(...)
    revenu_estime: float = Field(..., gt=0)
    score_risque: int = Field(..., ge=1, le=100)
    historique_incidents: int = Field(..., ge=0)

    # Dette
    montant_impaye: float = Field(..., gt=0)
    anciennete_impaye_jours: int = Field(..., ge=0)
    nombre_echeances_impayees: int = Field(..., ge=0)

    # Procédure (Les acteurs sont gérés automatiquement si la procédure prédite est judiciaire)
    huissier_id: str = Field(...)
    nombre_evenements: int = Field(..., ge=0)
    nombre_retards: int = Field(..., ge=0)
    derniere_action_age_jours: int = Field(..., ge=0)

    # Dates
    date_ouverture: date = Field(...)
    date_mise_a_jour: date = Field(...)

    @field_validator("date_mise_a_jour")
    @classmethod
    def date_mise_a_jour_after_ouverture(cls, v, info):
        if "date_ouverture" in info.data and v < info.data["date_ouverture"]:
            raise ValueError("date_mise_a_jour ne peut pas être antérieure à date_ouverture.")
        return v


class PredictionResponse(BaseModel):
    """
    Réponse complète renvoyée au client avec les résultats des 4 modèles ML
    + le scoring avocat/huissier calculé depuis la base de données.
    """
    # Résultats des 5 modèles ML
    procedure_recommandee: str = Field(..., description="Procédure recommandée par l'IA (Amiable / Judiciaire)")
    statut_final_predit: str = Field(..., description="Statut prédit (Recouvré / En cours / Échec)")
    probabilite_recouvrement: float = Field(..., description="Probabilité de recouvrement (0.0 - 1.0)")
    delai_estime_jours: float = Field(..., description="Durée estimée de la procédure en jours")
    next_best_action: str = Field(..., description="Action recommandée par l'IA")
    cluster_segment_id: int = Field(..., description="Segment du dossier (clustering KMeans)")

    # Scoring calculé depuis PostgreSQL
    score_avocat: float = Field(..., description="Score de performance de l'avocat")
    score_huissier: float = Field(..., description="Score de performance de l'huissier")

    # Métriques calculées automatiquement (pour transparence)
    acteur_taux_succes: float = Field(..., description="Taux de succès historique de l'avocat (calculé)")
    acteur_delai_moyen: float = Field(..., description="Délai moyen de l'avocat (calculé)")


class HistoryRecordResponse(BaseModel):
    """Schéma de sortie pour un enregistrement de l'historique."""
    id: int
    dossier_id: str
    client_segment: str
    montant_impaye: float
    statut_predit: str
    probabilite_recouvrement: float
    next_best_action: str
    cluster_segment_id: int
    score_avocat: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    models_loaded: bool
    version: str
