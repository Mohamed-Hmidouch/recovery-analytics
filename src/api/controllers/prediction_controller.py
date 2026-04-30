"""
Controller Layer — Routeur FastAPI (contrôleur 'stupide').
Gère uniquement le flux HTTP : réception, délégation au Service, gestion des erreurs.
Protégé par Rate Limiting (slowapi) et API Key Authentication (Depends).
Équivalent d'un @RestController + @PreAuthorize dans Spring Boot.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, Depends, Request, status

from src.api.schemas.dossier import DossierRequest, PredictionResponse, HealthResponse, HistoryRecordResponse
from src.api.services.prediction_service import PredictionService
from src.api.core.model_manager import model_manager
from src.api.core.security import limiter, verify_api_key
from src.api.db.models import PredictionHistory

from sqlalchemy.orm import Session
from src.api.db.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Recouvrement"])


@router.post(
    "/predict/recouvrement",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédiction de recouvrement",
    description="Reçoit un dossier de recouvrement et retourne la prédiction du statut final, "
                "la probabilité de recouvrement, le délai estimé, le score avocat, et le segment (KMeans). "
                "Requiert un header X-API-Key valide. Limité à 5 requêtes/minute/IP.",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("5/minute")
async def predict_recouvrement(request: Request, dossier: DossierRequest, db: Session = Depends(get_db)):
    """
    Endpoint principal de prédiction.
    Sécurité appliquée :
      - Couche 1 : Payload Guard Middleware (anti-JSON bomb, < 1 MB)
      - Couche 2 : Rate Limiting slowapi (5 req/min/IP)
      - Couche 3 : API Key Authentication (header X-API-Key)
      - Couche 4 : Validation Pydantic (DossierRequest)
      - Couche 5 : Try/Except (pas de stack trace exposée)
    """
    try:
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Les modèles ML ne sont pas encore chargés. Réessayez dans quelques secondes.",
            )
        result = PredictionService.predict(dossier, db)
        return result

    except HTTPException:
        # Laisse passer les HTTPException déjà formatées (comme le 503 ci-dessus)
        raise

    except Exception as e:
        # Sécurité : on log le détail côté serveur mais on ne renvoie RIEN au client
        logger.error(f"Erreur interne lors de la prédiction : {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors du traitement de la prédiction.",
        )


@router.get(
    "/history",
    response_model=List[HistoryRecordResponse],
    summary="Historique des prédictions",
    description="Renvoie les 50 dernières prédictions historisées dans PostgreSQL.",
    dependencies=[Depends(verify_api_key)],
)
async def get_prediction_history(db: Session = Depends(get_db), limit: int = 50):
    """
    Récupère l'historique récent des prédictions.
    Protégé par API Key.
    """
    history = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
    return history


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Vérifie que l'API est opérationnelle et que les modèles sont chargés. "
                "Pas d'authentification requise (utilisé par les orchestrateurs K8s/Docker).",
)
async def health_check():
    """Endpoint de santé — pas protégé par API Key (intentionnel pour les probes K8s)."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        models_loaded=model_manager.is_loaded,
        version="1.0.0",
    )
