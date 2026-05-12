"""
LangChain Tools — Fonctions que l'agent Gemini peut appeler en autonomie.

Tool 1 : predict_dossier     → appelle PredictionService (5 modèles Spark)
Tool 2 : query_history       → interroge prediction_history PostgreSQL
Tool 3 : get_segment_scoring → calcule le scoring par segment via ScoringService
"""

import logging
import json
import re
import uuid
from langchain_core.tools import tool

from src.api.db.database import SessionLocal
from src.api.db.models import PredictionHistory
from src.api.services.scoring_service import ScoringService

logger = logging.getLogger(__name__)


def _clean_arg(val) -> str:
    """Nettoie les arguments mal parsés par le ReAct agent.
    Exemple : client_segment='Retail' → 'Retail'  |  {'key': 'val'} → ''
    """
    if val is None:
        return ""
    s = str(val).strip()
    # cas : client_segment='Retail' ou key="Retail"
    m = re.search(r"""=\s*['"]?([A-Za-zÀ-ÿ_0-9]+)['"]?""", s)
    if m:
        return m.group(1)
    # cas : 'Retail' ou "Retail"
    s = s.strip("'\"")
    # cas : {} ou None ou vide
    if s in ("{}", "null", "none", "None", ""):
        return ""
    return s


@tool
def predict_dossier(
    dossier_id: str = "",
    client_segment: str = "Retail",
    revenu_estime: float = 20000.0,
    montant_impaye: float = 5000.0,
    historique_incidents: int = 1,
    anciennete_impaye_jours: int = 90,
    nombre_echeances_impayees: int = 3,
    date_ouverture: str = "2024-01-01",
    date_mise_a_jour: str = "2024-06-01",
    domiciliation_salaire: bool = False,
    anciennete_client_annees: int = 3,
    statut_matrimonial: str = "Celibataire",
    personnes_a_charge: int = 0,
    categorie_employeur: str = "Prive",
    type_contrat: str = "CDI",
    statut_logement: str = "Locataire",
) -> str:
    """
    Lance la prédiction complète des 5 modèles ML pour un dossier de recouvrement.
    Retourne : procédure recommandée, probabilité de recouvrement, statut prédit,
    délai estimé, next best action, cluster, score avocat.
    Utilise ce tool quand l'utilisateur veut simuler ou analyser un dossier précis.
    Tous les paramètres ont des valeurs par défaut — utilise-les si non fournis.
    """
    from src.api.schemas.dossier import DossierRequest
    from src.api.services.prediction_service import PredictionService

    # Mapping categorie_employeur si le banquier dit "indépendant" ou "fonctionnaire"
    emp_map = {"independant": "Independant", "fonctionnaire": "Etat", "etat": "Etat",
               "prive": "Prive", "sans emploi": "Sans_emploi", "sans_emploi": "Sans_emploi"}
    contrat_map = {"fonctionnaire": "Fonctionnaire", "cdi": "CDI", "cdd": "CDD",
                   "interim": "Interim", "sans contrat": "Sans_contrat", "sans_contrat": "Sans_contrat"}
    seg_map = {"retail": "Retail", "professionnel": "Professionnel", "corporate": "Corporate"}

    seg = seg_map.get(str(client_segment).lower(), client_segment)
    emp = emp_map.get(str(categorie_employeur).lower(), categorie_employeur)
    contrat = contrat_map.get(str(type_contrat).lower(), type_contrat)

    db = SessionLocal()
    try:
        request = DossierRequest(
            dossier_id=dossier_id or f"AGENT-{uuid.uuid4().hex[:8]}",
            procedure_id=f"PROC-AGENT-{uuid.uuid4().hex[:8]}",
            client_segment=seg,
            revenu_estime=float(revenu_estime),
            montant_impaye=float(montant_impaye),
            historique_incidents=int(historique_incidents),
            anciennete_impaye_jours=int(anciennete_impaye_jours),
            nombre_echeances_impayees=int(nombre_echeances_impayees),
            date_ouverture=date_ouverture,
            date_mise_a_jour=date_mise_a_jour,
            domiciliation_salaire=bool(domiciliation_salaire),
            anciennete_client_annees=int(anciennete_client_annees),
            statut_matrimonial=statut_matrimonial,
            personnes_a_charge=int(personnes_a_charge),
            categorie_employeur=emp,
            type_contrat=contrat,
            statut_logement=statut_logement,
        )
        result = PredictionService.predict(request, db)
        return json.dumps({
            "meilleure_procedure": result.meilleure_procedure,
            "probabilite_recouvrement": f"{result.taux_de_succes * 100:.1f}%",
            "statut_final_predit": result.statut_final_predit,
            "delai_estime_jours": f"{result.delai_estime_jours:.0f} jours",
            "prochaine_action_recommandee": result.prochaine_action_recommandee,
            "cluster_segment_id": result.cluster_segment_id,
            "score_avocat": round(result.score_avocat, 1),
            "score_huissier": round(result.score_huissier, 1),
            "taux_historique_segment": f"{result.acteur_taux_succes * 100:.1f}%",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"predict_dossier tool error: {e}", exc_info=True)
        return f"Erreur lors de la prédiction : {str(e)}"
    finally:
        db.close()


@tool
def query_history(client_segment: str = "", limit: int = 20) -> str:
    """
    Interroge l'historique des prédictions dans PostgreSQL.
    Filtre optionnel par client_segment (Retail, Professionnel, Corporate).
    Retourne les statistiques agrégées : nombre de dossiers, taux de succès moyen,
    délai moyen, distribution des statuts et des procédures.
    Utilise ce tool quand l'utilisateur pose des questions sur l'historique ou les performances.
    Pour tous les segments, laisse client_segment vide.
    """
    db = SessionLocal()
    try:
        seg = _clean_arg(client_segment)
        query = db.query(PredictionHistory)
        if seg:
            query = query.filter(PredictionHistory.client_segment == seg)
        records = query.order_by(PredictionHistory.created_at.desc()).limit(limit).all()

        # Fallback : si segment non trouvé, retourne tout
        if not records and seg:
            records = db.query(PredictionHistory).order_by(
                PredictionHistory.created_at.desc()
            ).limit(limit).all()
            seg = f"tous ('{seg}' non trouvé en base)"

        if not records:
            return "Aucun dossier en base de données pour le moment."

        total = len(records)
        statuts: dict = {}
        procedures: dict = {}
        segments: dict = {}
        taux_sum = 0.0
        delai_sum = 0.0

        for r in records:
            statuts[r.statut_predit] = statuts.get(r.statut_predit, 0) + 1
            procedures[r.type_procedure] = procedures.get(r.type_procedure, 0) + 1
            segments[r.client_segment] = segments.get(r.client_segment, 0) + 1
            taux_sum += r.probabilite_recouvrement or 0
            delai_sum += r.delai_estime_jours or 0

        return json.dumps({
            "segment_filtre": seg or "tous",
            "nombre_dossiers_analyses": total,
            "probabilite_recouvrement_moyenne": f"{(taux_sum / total) * 100:.1f}%",
            "delai_moyen_estime_jours": round(delai_sum / total, 1),
            "distribution_statuts": statuts,
            "distribution_procedures": procedures,
            "distribution_segments": segments,
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"query_history tool error: {e}", exc_info=True)
        return f"Erreur lors de la requête historique : {str(e)}"
    finally:
        db.close()


@tool
def get_segment_scoring(client_segment: str = "Retail", type_procedure: str = "Amiable") -> str:
    """
    Calcule le scoring agrégé (taux de succès, délai moyen, score avocat/huissier)
    pour un segment client donné depuis l'historique PostgreSQL.
    client_segment : Retail, Professionnel ou Corporate.
    type_procedure : Amiable ou Judiciaire.
    Utilise ce tool quand l'utilisateur demande les performances par segment ou avocat.
    """
    db = SessionLocal()
    try:
        seg = _clean_arg(client_segment) or "Retail"
        proc = _clean_arg(type_procedure) or "Amiable"

        seg_metrics = ScoringService.compute_segment_metrics(db, seg)
        trib_metrics = ScoringService.compute_tribunal_metrics(db, seg, proc)
        proc_metrics = ScoringService.compute_procedure_metrics(db, seg, proc)
        score_avocat = ScoringService.compute_score_avocat(
            seg_metrics["acteur_taux_succes"], seg_metrics["acteur_delai_moyen"]
        )
        score_huissier = ScoringService.compute_score_huissier(db, seg)

        return json.dumps({
            "segment": seg,
            "type_procedure": proc,
            "taux_succes_segment": f"{seg_metrics['acteur_taux_succes'] * 100:.1f}%",
            "delai_moyen_jours": seg_metrics["acteur_delai_moyen"],
            "score_avocat_sur_100": round(score_avocat, 1),
            "score_huissier_sur_100": round(score_huissier, 1),
            "taux_succes_procedure": f"{proc_metrics['procedure_taux_succes'] * 100:.1f}%",
            "delai_tribunal_jours": trib_metrics["tribunal_delai_moyen"],
            "interpretation": "Score > 70 = excellent | 50-70 = correct | < 50 = à améliorer",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"get_segment_scoring tool error: {e}", exc_info=True)
        return f"Erreur scoring : {str(e)}"
    finally:
        db.close()
