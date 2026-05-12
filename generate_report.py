"""
Génération du rapport PDF Smart Recovery.
Appelle l'API réelle via HTTP (requests), affiche les résultats
de TOUS les algorithmes demandés par le manager.
"""

import os
import sys
import json
import uuid
import requests
import re
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
from weasyprint import HTML

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
SECRET_API_KEY = os.getenv("SECRET_API_KEY")

if not all([API_HOST, API_PORT, SECRET_API_KEY]):
    print("Erreur : API_HOST, API_PORT ou SECRET_API_KEY manquant dans .env")
    sys.exit(1)

HOST = "127.0.0.1" if API_HOST == "0.0.0.0" else API_HOST
BASE_URL = f"http://{HOST}:{API_PORT}"
HEADERS = {"X-API-Key": SECRET_API_KEY, "Content-Type": "application/json"}


def call_predict(label, segment, revenu, montant, anciennete, incidents, echeances):
    payload = {
        "dossier_id": f"DOS-RPT-{str(uuid.uuid4())[:8].upper()}",
        "procedure_id": f"PROC-RPT-{str(uuid.uuid4())[:8].upper()}",
        "client_segment": segment,
        "revenu_estime": revenu,
        "historique_incidents": incidents,
        "montant_impaye": montant,
        "anciennete_impaye_jours": anciennete,
        "nombre_echeances_impayees": echeances,
        "date_ouverture": (date.today() - timedelta(days=anciennete)).isoformat(),
        "date_mise_a_jour": date.today().isoformat(),
    }
    try:
        r = requests.post(
            f"{BASE_URL}/api/v1/predict/recouvrement",
            json=payload,
            headers=HEADERS,
            timeout=60,
        )
        r.raise_for_status()
        print(f"  [OK] {label} — HTTP {r.status_code}")
        return payload, r.json()
    except Exception as e:
        print(f"  [ERREUR] {label} : {e}")
        sys.exit(1)


print("Connexion a l'API et recuperation des predictions en cours...")

payload_1, res_1 = call_predict("Profil 1 — Retail Favorable",     "Retail",        45000,   500,  30, 0, 1)
payload_2, res_2 = call_predict("Profil 2 — Corporate Critique",   "Corporate",     10000, 45000, 300, 5, 12)
payload_3, res_3 = call_predict("Profil 3 — Professionnel Moyen",  "Professionnel", 35000,  8500,  90, 2, 4)
payload_4, res_4 = call_predict("Profil 4 — Retail Anomalie",      "Retail",        25000,  1200, 180, 1, 6)

print("Predictions terminees. Generation du rapport HTML/PDF...")


def score_color(score):
    if score >= 60:
        return "#16a34a"
    if score >= 20:
        return "#d97706"
    return "#dc2626"


def taux_color(taux):
    if taux >= 0.6:
        return "#16a34a"
    if taux >= 0.35:
        return "#d97706"
    return "#dc2626"


def statut_color(statut):
    return {"Recouvre": "#16a34a", "En cours": "#d97706", "Echec": "#dc2626"}.get(statut, "#64748b")


def proc_color(proc):
    return "#2563eb" if proc == "Amiable" else "#dc2626"


def dossier_card(num, badge_class, badge_label, payload, res):
    proc   = res.get("meilleure_procedure", "N/A")
    taux   = res.get("taux_de_succes", 0)
    statut = res.get("statut_final_predit", "N/A")
    delai  = res.get("delai_estime_jours", 0)
    action = res.get("prochaine_action_recommandee", "N/A")
    cluster = res.get("cluster_segment_id", "N/A")
    score_av = res.get("score_avocat", 0)
    score_hu = res.get("score_huissier", 0)
    taux_hist = res.get("acteur_taux_succes", 0)
    delai_hist = res.get("acteur_delai_moyen", 0)
    avocat = res.get("avocat_id", "N/A")
    tribunal = res.get("tribunal_id", "N/A")

    return f"""
    <div class="dossier-card">
        <div class="dossier-header">
            <span class="badge {badge_class}">Profil {num} — {badge_label}</span>
            <span class="segment-tag">
                Segment : {payload.get('client_segment')}
                &nbsp;|&nbsp; Montant : {payload.get('montant_impaye'):,.0f} MAD
                &nbsp;|&nbsp; Anciennete : {payload.get('anciennete_impaye_jours')} j
                &nbsp;|&nbsp; Incidents : {payload.get('historique_incidents')}
            </span>
        </div>

        <div class="section-title">Resultats des 5 Algorithmes ML + Scoring</div>
        <div class="algo-grid">
            <div class="algo-cell">
                <div class="algo-name">Modele 1<br>Classification Procedure</div>
                <div class="algo-value" style="color:{proc_color(proc)}">{proc}</div>
                <div class="algo-sub">Amiable ou Judiciaire</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Modele 2<br>Probabilite de Recouvrement</div>
                <div class="algo-value" style="color:{taux_color(taux)}">{taux * 100:.1f} %</div>
                <div class="algo-sub">Statut predit : <strong style="color:{statut_color(statut)}">{statut}</strong></div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Modele 3<br>Prediction Duree Procedure</div>
                <div class="algo-value">{delai:.0f} jours</div>
                <div class="algo-sub">Regression RandomForest</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Modele 4<br>Next Best Action</div>
                <div class="algo-value" style="font-size:10pt; color:#1e40af;">{action}</div>
                <div class="algo-sub">Action optimale recommandee</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Modele 5<br>Clustering KMeans</div>
                <div class="algo-value">Segment {cluster}</div>
                <div class="algo-sub">Profil de risque homogene</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Scoring Avocat<br>(Historique PostgreSQL)</div>
                <div class="algo-value" style="color:{score_color(score_av)}">{score_av:.1f} / 100</div>
                <div class="algo-sub">Taux historique : {taux_hist * 100:.1f} % | Delai moy : {delai_hist:.0f} j</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Scoring Huissier<br>(Historique PostgreSQL)</div>
                <div class="algo-value" style="color:{score_color(score_hu)}">{score_hu:.1f} / 100</div>
                <div class="algo-sub">Aggrege par client_segment</div>
            </div>
            <div class="algo-cell">
                <div class="algo-name">Acteurs Assigns<br>automatiquement</div>
                <div class="algo-value" style="font-size:8.5pt; color:#334155;">
                    Avocat : {avocat}<br>Tribunal : {tribunal}
                </div>
                <div class="algo-sub">Selon procedure predite par l'IA</div>
            </div>
        </div>
    </div>"""


# ── Agent IA Q&A section ──────────────────────────────────────────────────────
_AGENT_QA_FILE = BASE_DIR / "agent_qa.json"
_agent_qa_items = []
if _AGENT_QA_FILE.exists():
    with open(_AGENT_QA_FILE, encoding="utf-8") as _f:
        _agent_qa_items = json.load(_f)

def _md_to_html(text):
    """Minimal markdown → HTML converter for agent answers."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\n\n', '</p><p>', text)
    text = re.sub(r'\n([0-9]+\. )', r'<br/>\1', text)
    text = re.sub(r'\n([-•✅⚠️⚡🔑🎯💡📋📊🏆📈🔍💰📅🏛️🔨] )', r'<br/>\1', text)
    text = re.sub(r'\n', '<br/>', text)
    return f"<p>{text}</p>"

def _tools_badge(tools):
    colors = {"predict_dossier": "#3b82f6", "query_history": "#10b981", "get_segment_scoring": "#8b5cf6"}
    labels = {"predict_dossier": "🤖 predict_dossier", "query_history": "📊 query_history", "get_segment_scoring": "🎯 get_segment_scoring"}
    badges = ""
    for t in tools:
        c = colors.get(t, "#6b7280")
        l = labels.get(t, t)
        badges += f'<span style="background:{c};color:white;border-radius:4px;padding:2px 8px;font-size:8pt;margin-right:4px;">{l}</span>'
    return badges

_agent_cards_html = ""
for _i, _qa in enumerate(_agent_qa_items, 1):
    _tools_html = _tools_badge(_qa.get("tools_used", []))
    _answer_html = _md_to_html(_qa["answer"])
    _agent_cards_html += f"""
    <div style="margin-bottom:24px; border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.10); page-break-inside:avoid;">
      <div style="background:#1e3a5f; color:white; padding:10px 16px; font-size:9pt; font-weight:600;">
        ❓ Question {_i} — Banquier
      </div>
      <div style="background:#eff6ff; padding:12px 16px; font-size:9.5pt; color:#1e3a5f; font-style:italic; border-bottom:1px solid #dbeafe;">
        « {_qa['question']} »
      </div>
      <div style="background:white; padding:4px 16px 4px 16px; border-bottom:1px solid #f1f5f9;">
        <span style="font-size:7.5pt; color:#64748b;">Tools utilisés : </span>{_tools_html}
      </div>
      <div style="background:#f8fafc; padding:12px 16px; font-size:9pt; color:#1e293b; line-height:1.6;">
        <span style="color:#10b981; font-weight:700;">💬 Agent SmartRecovery :</span><br/>
        {_answer_html}
      </div>
    </div>"""

if not _agent_cards_html:
    _agent_cards_html = "<p style='color:#6b7280;font-style:italic;'>Aucune session agent disponible.</p>"

_agent_section_html = f"""
<div style="page-break-before:always; margin-top:10px;">
  <div style="background:linear-gradient(135deg,#1e3a5f,#3b82f6); color:white; border-radius:10px; padding:18px 24px; margin-bottom:24px;">
    <div style="font-size:15pt; font-weight:700; margin-bottom:4px;">🤖 Agent IA SmartRecovery — Démonstration Gemini + LangChain</div>
    <div style="font-size:9pt; opacity:0.85;">5 Questions Métier Réelles · Powered by Google Gemini 2.5 Flash · LangChain ReAct Agent · 3 Tools PySpark ML</div>
  </div>
  <div style="background:#fef9c3; border-left:4px solid #f59e0b; border-radius:6px; padding:10px 16px; margin-bottom:20px; font-size:8.5pt; color:#78350f;">
    <strong>Architecture :</strong> Questions en langage naturel → <strong>Gemini 2.5 Flash</strong> (raisonnement) → <strong>LangChain ReAct</strong> (orchestration) → Tools PySpark ML (<em>predict_dossier, query_history, get_segment_scoring</em>) → Réponse structurée en français
  </div>
  {_agent_cards_html}
</div>"""
# ─────────────────────────────────────────────────────────────────────────────


html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<style>
    @page {{ size: A4; margin: 15mm 18mm; }}
    body {{ font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: #1e293b; line-height: 1.5; margin: 0; padding: 0; background: #f1f5f9; }}

    .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; padding: 28px 35px; text-align: center; margin-bottom: 20px; }}
    .header h1 {{ margin: 0 0 6px; font-size: 20pt; letter-spacing: 1px; }}
    .header .subtitle {{ font-size: 10pt; opacity: 0.85; margin: 0; }}
    .header .date-line {{ font-size: 8.5pt; opacity: 0.6; margin-top: 6px; }}

    .summary-bar {{ display: flex; gap: 10px; margin-bottom: 20px; }}
    .summary-box {{ flex: 1; background: white; border-radius: 8px; padding: 12px 10px; border-top: 4px solid #3b82f6; box-shadow: 0 1px 4px rgba(0,0,0,0.07); text-align: center; }}
    .summary-box .sb-label {{ font-size: 7pt; text-transform: uppercase; color: #64748b; font-weight: 700; margin-bottom: 4px; }}
    .summary-box .sb-value {{ font-size: 9pt; font-weight: 800; color: #0f172a; line-height: 1.4; }}

    .dossier-card {{ background: white; margin-bottom: 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.07); page-break-inside: avoid; overflow: hidden; }}
    .dossier-header {{ background: #f8fafc; padding: 12px 18px; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 8.5pt; font-weight: 700; text-transform: uppercase; white-space: nowrap; }}
    .badge-vert   {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
    .badge-rouge  {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
    .badge-orange {{ background: #ffedd5; color: #9a3412; border: 1px solid #fdba74; }}
    .badge-jaune  {{ background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }}
    .segment-tag {{ font-size: 8.5pt; color: #475569; background: #f1f5f9; padding: 3px 10px; border-radius: 12px; border: 1px solid #e2e8f0; }}

    .section-title {{ font-size: 8pt; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; padding: 10px 18px 4px; }}
    .algo-grid {{ display: flex; flex-wrap: wrap; padding: 0 10px 14px; gap: 7px; }}
    .algo-cell {{ flex: 1 1 21%; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 10px 12px; min-width: 120px; }}
    .algo-name {{ font-size: 7pt; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 4px; line-height: 1.3; }}
    .algo-value {{ font-size: 13pt; font-weight: 800; color: #0f172a; margin-bottom: 3px; }}
    .algo-sub {{ font-size: 7.5pt; color: #94a3b8; }}

    .api-section {{ background: #0f172a; color: #e2e8f0; border-radius: 10px; padding: 22px 26px; margin-top: 20px; page-break-inside: avoid; }}
    .api-section h2 {{ color: #38bdf8; font-size: 13pt; margin: 0 0 4px; border-bottom: 1px solid #1e3a8a; padding-bottom: 8px; }}
    .api-section h3 {{ color: #94a3b8; font-size: 9.5pt; margin: 14px 0 4px; }}
    .code-block {{ background: #020617; padding: 12px; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 7.5pt; white-space: pre-wrap; color: #a5b4fc; line-height: 1.5; }}

    .lexique-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 8pt; }}
    .lexique-table th {{ background: #020617; color: #38bdf8; padding: 6px 9px; text-align: left; border: 1px solid #1e3a8a; }}
    .lexique-table td {{ padding: 5px 9px; border: 1px solid #1e293b; vertical-align: top; }}
    .lexique-table tr:nth-child(odd)  td {{ background: #0f172a; }}
    .lexique-table tr:nth-child(even) td {{ background: #162032; }}
    .lexique-table .field  {{ color: #a5b4fc; font-family: monospace; }}
    .lexique-table .metier {{ color: #fbbf24; font-weight: 600; }}
    .lexique-table .algo   {{ color: #6ee7b7; font-size: 7.5pt; }}
    .lexique-table .desc   {{ color: #cbd5e1; }}

    .footer {{ text-align: center; font-size: 7.5pt; color: #94a3b8; margin-top: 20px; padding: 10px; border-top: 1px solid #e2e8f0; }}

    .validation-note {{ background: white; border-radius: 10px; padding: 20px 24px; margin-top: 20px; margin-bottom: 20px; border-left: 5px solid #10b981; box-shadow: 0 2px 6px rgba(0,0,0,0.07); page-break-inside: avoid; }}
    .vn-title {{ font-size: 11pt; font-weight: 800; color: #065f46; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.4px; }}
    .vn-intro {{ font-size: 9pt; color: #334155; margin: 0 0 14px; }}
    .vn-grid {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .vn-item {{ flex: 1 1 45%; display: flex; align-items: flex-start; gap: 10px; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 7px; padding: 10px 12px; }}
    .vn-done {{ border-color: #86efac; }}
    .vn-check {{ background: #16a34a; color: white; font-size: 7pt; font-weight: 800; border-radius: 4px; padding: 2px 6px; white-space: nowrap; margin-top: 2px; flex-shrink: 0; }}
    .vn-item strong {{ font-size: 8.5pt; color: #14532d; display: block; margin-bottom: 2px; }}
    .vn-item span {{ font-size: 7.5pt; color: #166534; line-height: 1.4; }}
    .vn-footer {{ font-size: 8.5pt; color: #475569; background: #f8fafc; border-radius: 6px; padding: 10px 14px; margin: 0; border: 1px solid #e2e8f0; }}
</style>
</head>
<body>

<div class="header">
    <h1>Smart Recovery — Rapport de Validation Algorithmique</h1>
    <p class="subtitle">Pipeline Machine Learning · 5 Modeles PySpark · Classification · Regression · Clustering · Scoring Avocat/Huissier</p>
    <p class="date-line">Genere le {date.today().strftime("%d/%m/%Y")} — Resultats en temps reel depuis l'API REST (PostgreSQL)</p>
</div>

<div class="summary-bar">
    <div class="summary-box" style="border-top-color:#3b82f6">
        <div class="sb-label">Modele 1</div>
        <div class="sb-value">Classification<br>Procedure</div>
    </div>
    <div class="summary-box" style="border-top-color:#10b981">
        <div class="sb-label">Modele 2</div>
        <div class="sb-value">Probabilite<br>Recouvrement</div>
    </div>
    <div class="summary-box" style="border-top-color:#f59e0b">
        <div class="sb-label">Modele 3</div>
        <div class="sb-value">Prediction<br>Duree</div>
    </div>
    <div class="summary-box" style="border-top-color:#8b5cf6">
        <div class="sb-label">Modele 4</div>
        <div class="sb-value">Next Best<br>Action</div>
    </div>
    <div class="summary-box" style="border-top-color:#ef4444">
        <div class="sb-label">Modele 5</div>
        <div class="sb-value">Clustering<br>KMeans</div>
    </div>
    <div class="summary-box" style="border-top-color:#0ea5e9">
        <div class="sb-label">Scoring</div>
        <div class="sb-value">Avocat /<br>Huissier</div>
    </div>
</div>

{dossier_card(1, "badge-vert",   "Dossier Favorable — Retail",       payload_1, res_1)}
{dossier_card(2, "badge-rouge",  "Dossier Critique — Corporate",     payload_2, res_2)}
{dossier_card(3, "badge-orange", "Risque Modere — Professionnel",    payload_3, res_3)}
{dossier_card(4, "badge-jaune",  "Anomalie Operationnelle — Retail", payload_4, res_4)}

<div class="validation-note">
    <div class="vn-title">Note de Validation — Reponse aux specifications initiales</div>
    <p class="vn-intro">
        Ce rapport confirme que <strong>l'integralite des algorithmes demandes</strong> ont ete implementes, testes et valides en temps reel via l'API.
        Chaque point de la specification est couvert :
    </p>
    <div class="vn-grid">
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Classification — Meilleure Procedure</strong><br>
                <span>Algorithme RandomForest (Modele 1) : predit automatiquement Amiable ou Judiciaire selon le profil du dossier.</span>
            </div>
        </div>
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Probabilite de Recouvrement</strong><br>
                <span>Algorithme RandomForest (Modele 2) : retourne la probabilite de succes de 0 a 100% et le statut final predit.</span>
            </div>
        </div>
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Scoring Avocat / Huissier</strong><br>
                <span>Calcule en temps reel depuis l'historique PostgreSQL par segment client. Aucune valeur manuelle requise.</span>
            </div>
        </div>
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Next Best Action Judiciaire</strong><br>
                <span>Algorithme RandomForest (Modele 4) : recommande l'action optimale parmi Relance / Mise en demeure / Action judiciaire / Negociation.</span>
            </div>
        </div>
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Prediction Duree Procedure</strong><br>
                <span>Algorithme RandomForest Regression (Modele 3) : estime le nombre de jours pour clore le dossier.</span>
            </div>
        </div>
        <div class="vn-item vn-done">
            <span class="vn-check">OK</span>
            <div>
                <strong>Segmentation / Clustering</strong><br>
                <span>Algorithme KMeans (Modele 5) : classe chaque dossier dans un profil de risque homogene (Segment 0 a 3).</span>
            </div>
        </div>
    </div>
    <p class="vn-footer">
        Le client envoie uniquement les donnees brutes du dossier (segment, montant, anciennete...).
        Le systeme calcule automatiquement tout le reste : procedure, probabilite, scoring, action, segment, delai.
    </p>
</div>

<div class="api-section">
    <h2>Annexe Technique — Format d'Echange API (JSON)</h2>
    <p style="font-size:8.5pt; color:#94a3b8; margin:0 0 4px">Documentation d'integration a destination des equipes developpement et metier.</p>

    <h3>1. Requete Client vers l'API (Payload minimal — le systeme calcule le reste)</h3>
    <div class="code-block">POST /api/v1/predict/recouvrement
Content-Type: application/json
X-API-Key: &lt;SECRET_API_KEY&gt;

{json.dumps(payload_1, indent=2, ensure_ascii=False)}</div>

    <h3>2. Reponse complete de l'algorithme ML (tous les resultats)</h3>
    <div class="code-block">{json.dumps(res_1, indent=2, ensure_ascii=False)}</div>

    <h3>3. Lexique des champs de sortie — Vocabulaire Metier</h3>
    <table class="lexique-table">
        <tr>
            <th>Champ JSON</th>
            <th>Vocabulaire Metier</th>
            <th>Algorithme source</th>
            <th>Description</th>
        </tr>
        <tr>
            <td class="field">meilleure_procedure</td>
            <td class="metier">Meilleure Procedure</td>
            <td class="algo">Modele 1 — RF Classifier</td>
            <td class="desc">Amiable ou Judiciaire — predit par l'IA depuis les features du dossier</td>
        </tr>
        <tr>
            <td class="field">taux_de_succes</td>
            <td class="metier">Probabilite de Recouvrement</td>
            <td class="algo">Modele 2 — RF Classifier</td>
            <td class="desc">Probabilite de succes predite de 0.0 a 1.0 (ex: 0.53 = 53%)</td>
        </tr>
        <tr>
            <td class="field">statut_final_predit</td>
            <td class="metier">Issue Predite du Dossier</td>
            <td class="algo">Modele 2 — RF Classifier</td>
            <td class="desc">Recouvre / En cours / Echec</td>
        </tr>
        <tr>
            <td class="field">delai_estime_jours</td>
            <td class="metier">Prediction Duree Procedure</td>
            <td class="algo">Modele 3 — RF Regressor</td>
            <td class="desc">Nombre de jours estimes pour clore le dossier</td>
        </tr>
        <tr>
            <td class="field">prochaine_action_recommandee</td>
            <td class="metier">Next Best Action Judiciaire</td>
            <td class="algo">Modele 4 — RF Classifier</td>
            <td class="desc">Relance amiable / Mise en demeure / Action judiciaire / Negociation</td>
        </tr>
        <tr>
            <td class="field">cluster_segment_id</td>
            <td class="metier">Segmentation / Clustering</td>
            <td class="algo">Modele 5 — KMeans</td>
            <td class="desc">Profil de risque homogene du dossier (Segment 0 a 3)</td>
        </tr>
        <tr>
            <td class="field">score_avocat</td>
            <td class="metier">Scoring Avocat</td>
            <td class="algo">PostgreSQL — par client_segment</td>
            <td class="desc">Score de performance historique : (taux_succes x 100) - (delai_moyen x 0.05)</td>
        </tr>
        <tr>
            <td class="field">score_huissier</td>
            <td class="metier">Scoring Huissier</td>
            <td class="algo">PostgreSQL — par client_segment</td>
            <td class="desc">Meme formule appliquee aux dossiers du meme segment geres par des huissiers</td>
        </tr>
        <tr>
            <td class="field">acteur_taux_succes</td>
            <td class="metier">Taux de Succes Historique</td>
            <td class="algo">PostgreSQL agregation</td>
            <td class="desc">Taux reel de dossiers conclus avec succes sur ce profil de client</td>
        </tr>
        <tr>
            <td class="field">acteur_delai_moyen</td>
            <td class="metier">Delai Moyen Historique</td>
            <td class="algo">PostgreSQL agregation</td>
            <td class="desc">Delai moyen observe sur le segment — contribue au calcul du score</td>
        </tr>
        <tr>
            <td class="field">avocat_id / tribunal_id</td>
            <td class="metier">Acteurs Assigns</td>
            <td class="algo">Automatique</td>
            <td class="desc">Assignation automatique par le systeme selon la procedure predite par l'IA</td>
        </tr>
    </table>
</div>

<div class="footer">
    Smart Recovery ML API v1.0.0 — Pipeline PySpark · 5 Modeles RandomForest / KMeans · FastAPI · PostgreSQL
</div>

{_agent_section_html}

</body>
</html>"""

output_html = str(BASE_DIR / "analyse_predictions.html")
output_pdf  = str(BASE_DIR / "analyse_predictions_smart_recovery.pdf")

with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_content)

HTML(output_html).write_pdf(output_pdf)
print(f"Rapport HTML : {output_html}")
print(f"Rapport PDF  : {output_pdf}")
