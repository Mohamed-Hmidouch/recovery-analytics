"""
Core — Security Layer (Middlewares & Authentication).
Implémente les 3 couches de sécurité défensives :
  1. PayloadGuardMiddleware  → Anti-JSON Bomb (rejette les payloads > 1 MB)
  2. Rate Limiting (slowapi) → Anti-DDoS (5 req/min/IP sur /predict)
  3. API Key Auth (Depends)  → Authentification par header X-API-Key

Équivalent des @WebFilter / OncePerRequestFilter de Spring Security.
"""

import os
import secrets
import logging
from fastapi import Request, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

# ===========================================================================
# 1. MIDDLEWARE ANTI-JSON BOMB (Protection Payload)
# ===========================================================================
MAX_PAYLOAD_BYTES = 1 * 1024 * 1024  # 1 MB

class PayloadGuardMiddleware(BaseHTTPMiddleware):
    """
    Intercepte TOUTES les requêtes entrantes.
    Rejette immédiatement celles dont le Content-Length dépasse MAX_PAYLOAD_BYTES.
    Cela empêche un attaquant d'envoyer un JSON de 50 MB pour crasher la RAM.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_PAYLOAD_BYTES:
            logger.warning(
                f"SECURITY — Payload trop volumineux rejeté : {content_length} bytes "
                f"depuis {request.client.host}"
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "error_code": "PAYLOAD_TOO_LARGE",
                    "message": "Les données envoyées sont trop volumineuses. Pour des raisons de performance, la limite est fixée à 1 Mo.",
                    "details": []
                },
            )
        return await call_next(request)


# ===========================================================================
# 2. RATE LIMITER (Anti-DDoS / Brute Force)
# ===========================================================================
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handler personnalisé pour les erreurs de rate limit — pas de stack trace."""
    logger.warning(f"SECURITY — Rate limit dépassé pour {request.client.host} sur {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "success": False,
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": "Vous avez envoyé trop de requêtes en peu de temps. Veuillez patienter une minute avant de réessayer.",
            "details": []
        },
    )


# ===========================================================================
# 3. AUTHENTIFICATION PAR API KEY
# ===========================================================================
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    Dépendance FastAPI (Depends) qui vérifie le header X-API-Key.
    Utilise secrets.compare_digest pour une comparaison en temps constant,
    ce qui empêche les attaques par timing side-channel.
    """
    expected_key = os.getenv("SECRET_API_KEY")

    if not expected_key:
        logger.error("SECURITY — SECRET_API_KEY non définie dans .env ! L'API est non protégée.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration serveur invalide.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Accès refusé : la clé d'authentification est manquante. Veuillez fournir le header 'X-API-Key'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key, expected_key):
        logger.warning(f"SECURITY — Tentative d'accès avec une API Key invalide.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé : la clé d'API fournie est incorrecte ou a expiré.",
        )

    return api_key
