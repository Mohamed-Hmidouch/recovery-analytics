"""
Controller Layer — Routeur FastAPI (contrôleur 'stupide').
Gère uniquement le flux HTTP : réception, délégation au Service, gestion des erreurs.
Équivalent d'un @RestController dans Spring Boot.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from src.api.schemas.dossier import DossierRequest, PredictionResponse, HealthResponse
from src.api.services.prediction_service import PredictionService
from src.api.core.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Recouvrement"])


@router.post(
    "/predict/recouvrement",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédiction de recouvrement",
    description="Reçoit un dossier de recouvrement et retourne la prédiction du statut final, "
                "la probabilité de recouvrement, le délai estimé et le score avocat.",
)
async def predict_recouvrement(dossier: DossierRequest):
    """
    Endpoint principal de prédiction.
    La validation des entrées est assurée par le schéma Pydantic (DossierRequest).
    Les erreurs internes sont capturées et renvoyées sous forme de réponse HTTP 500 générique
    pour ne pas fuiter de stack trace au client (Security Best Practice).
    """
    try:
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Les modèles ML ne sont pas encore chargés. Réessayez dans quelques secondes.",
            )
        result = PredictionService.predict(dossier)
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
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Vérifie que l'API est opérationnelle et que les modèles sont chargés.",
)
async def health_check():
    """Endpoint de santé — utile pour les orchestrateurs (Docker, K8s)."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        models_loaded=model_manager.is_loaded,
        version="1.0.0",
    )
