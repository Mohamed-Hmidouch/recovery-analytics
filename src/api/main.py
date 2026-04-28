"""
Main — Point d'entrée de l'application FastAPI.
Assemble les routeurs, configure les middlewares de sécurité (CORS, Payload Guard, Rate Limit),
et gère le cycle de vie de l'application (lifespan) pour charger/décharger les modèles.
Équivalent de la classe @SpringBootApplication + SecurityFilterChain.
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from slowapi.errors import RateLimitExceeded

from src.api.core.model_manager import model_manager
from src.api.core.security import PayloadGuardMiddleware, limiter, rate_limit_exceeded_handler
from src.api.controllers.prediction_controller import router as prediction_router

# Charger les variables d'environnement depuis le .env à la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# Configuration du logging centralisée
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# -- Lifespan : gestion du cycle de vie (Startup / Shutdown) --
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge les modèles ML au démarrage et libère les ressources à l'arrêt.
    C'est l'équivalent du @PostConstruct / @PreDestroy de Spring.
    """
    logger.info("=== DÉMARRAGE DE L'API SmartRecovery ===")
    try:
        model_manager.startup()
        logger.info("Tous les modèles sont chargés. L'API est prête à recevoir des requêtes.")
    except Exception as e:
        logger.critical(f"Échec du chargement des modèles : {e}", exc_info=True)
        # On laisse l'API démarrer en mode dégradé pour que le /health retourne 'degraded'

    yield  # L'application tourne ici

    logger.info("=== ARRÊT DE L'API SmartRecovery ===")
    model_manager.shutdown()


# -- Création de l'application --
app = FastAPI(
    title="SmartRecovery ML API",
    description=(
        "API REST pour la prédiction de recouvrement de créances. "
        "Expose des modèles PySpark ML (RandomForest) via des endpoints sécurisés. "
        "Protégée par Rate Limiting, Payload Guard et API Key Authentication."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ===========================================================================
# COUCHE 1 : Middleware Anti-JSON Bomb (intercepte AVANT tout traitement)
# ===========================================================================
app.add_middleware(PayloadGuardMiddleware)

# ===========================================================================
# COUCHE 2 : Rate Limiter (slowapi)
# ===========================================================================
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ===========================================================================
# Middleware CORS
# ===========================================================================
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -- Enregistrement des routeurs --
app.include_router(prediction_router)


# -- Root endpoint (optionnel, pas protégé) --
@app.get("/", tags=["Root"])
async def root():
    return {
        "application": "SmartRecovery ML API",
        "version": "1.0.0",
        "docs": "/docs",
    }
