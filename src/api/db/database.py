import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# Charger explicitement le .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    logger.error("La variable DATABASE_URL n'est pas définie dans l'environnement ou le fichier .env.")
    raise ValueError("DATABASE_URL manquant.")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    # pool_pre_ping=True est recommandé pour PostgreSQL pour vérifier si la connexion est active
    pool_pre_ping=True 
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Générateur de session DB pour l'injection de dépendances (Depends) dans FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
