from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from src.api.db.database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    # Identifiants
    id = Column(Integer, primary_key=True, index=True)
    feature_id = Column(String, unique=True, index=True, nullable=False)
    dossier_id = Column(String, index=True, nullable=False)
    procedure_id = Column(String, index=True, nullable=False)

    # Client
    client_segment = Column(String, nullable=False)
    revenu_estime = Column(Float, nullable=False)
    score_risque = Column(Integer, nullable=False)
    historique_incidents = Column(Integer, nullable=False)

    # Dette
    montant_impaye = Column(Float, nullable=False)
    anciennete_impaye_jours = Column(Integer, nullable=False)
    nombre_echeances_impayees = Column(Integer, nullable=False)

    # Procédure
    type_procedure = Column(String, nullable=False)
    tribunal_id = Column(String, nullable=False)
    avocat_id = Column(String, nullable=False)
    huissier_id = Column(String, nullable=False)
    nombre_evenements = Column(Integer, nullable=False)
    nombre_retards = Column(Integer, nullable=False)
    derniere_action_age_jours = Column(Integer, nullable=False)

    # Performance Historique
    acteur_taux_succes = Column(Float, nullable=False)
    acteur_delai_moyen = Column(Float, nullable=False)
    tribunal_delai_moyen = Column(Float, nullable=False)
    procedure_taux_succes = Column(Float, nullable=False)

    # ML Predictions & Clustering
    cluster_segment_id = Column(Integer, nullable=False)  # Segment KMeans
    statut_predit = Column(String, nullable=False)
    probabilite_recouvrement = Column(Float, nullable=False)
    delai_estime_jours = Column(Float, nullable=False)
    score_avocat = Column(Float, nullable=False)
    next_best_action = Column(String, nullable=False)  # Recommendation IA

    # Outcome (Cibles réelles — remplies plus tard via feedback ou batch)
    montant_recouvre_final = Column(Float, nullable=True)
    delai_final_jours = Column(Integer, nullable=True)
    statut_final = Column(String, nullable=True)

    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
