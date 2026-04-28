"""
Schemas Pydantic (DTOs) — Première barrière de sécurité (Input Validation).
Équivalent des classes DTO/Record en Spring Boot.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import date


class DossierRequest(BaseModel):
    """
    Schéma de validation stricte des données d'entrée pour une prédiction.
    Chaque champ est contraint par son type, ses bornes et sa sémantique métier.
    """
    client_segment: Literal["Retail", "Professionnel", "Corporate"] = Field(
        ...,
        description="Segment client. Valeurs autorisées : Retail, Professionnel, Corporate."
    )
    revenu_estime: float = Field(
        ..., gt=0,
        description="Revenu estimé du débiteur en MAD. Doit être strictement positif."
    )
    score_risque: int = Field(
        ..., ge=1, le=100,
        description="Score de risque du dossier, entier entre 1 et 100."
    )
    montant_impaye: float = Field(
        ..., gt=0,
        description="Montant impayé en MAD. Doit être strictement positif."
    )
    type_procedure: Literal["Amiable", "Judiciaire"] = Field(
        ...,
        description="Type de procédure engagée. Valeurs autorisées : Amiable, Judiciaire."
    )
    acteur_taux_succes: float = Field(
        ..., ge=0.0, le=1.0,
        description="Taux de succès historique de l'acteur assigné, entre 0.0 et 1.0."
    )
    date_ouverture: date = Field(
        ...,
        description="Date d'ouverture du dossier (format ISO : YYYY-MM-DD)."
    )
    date_mise_a_jour: date = Field(
        ...,
        description="Date de dernière mise à jour du dossier (format ISO : YYYY-MM-DD)."
    )

    @field_validator("date_mise_a_jour")
    @classmethod
    def date_mise_a_jour_after_ouverture(cls, v, info):
        """La date de mise à jour doit être postérieure ou égale à la date d'ouverture."""
        if "date_ouverture" in info.data and v < info.data["date_ouverture"]:
            raise ValueError("date_mise_a_jour ne peut pas être antérieure à date_ouverture.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "client_segment": "Retail",
                    "revenu_estime": 45000.0,
                    "score_risque": 62,
                    "montant_impaye": 8500.50,
                    "type_procedure": "Amiable",
                    "acteur_taux_succes": 0.72,
                    "date_ouverture": "2024-03-15",
                    "date_mise_a_jour": "2024-09-20"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """
    Schéma de la réponse de prédiction renvoyée au client.
    Aucun détail interne (stack trace, nom de modèle, chemin FS) n'est exposé.
    """
    statut_final_predit: str = Field(
        ...,
        description="Statut final prédit par le classifier (Recouvré, Échec, En cours)."
    )
    probabilite_recouvrement: float = Field(
        ...,
        description="Probabilité de recouvrement (entre 0.0 et 1.0)."
    )
    delai_estime_jours: float = Field(
        ...,
        description="Délai estimé de résolution du dossier en jours."
    )
    score_avocat: float = Field(
        ...,
        description="Score composite de l'acteur/avocat calculé à partir de ses performances."
    )


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    models_loaded: bool
    version: str
