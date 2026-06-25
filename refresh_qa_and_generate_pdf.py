"""
Script de mise à jour Q&A réel + génération PDF chatbot.

1. Remplace les 4 questions query_history (Q2, Q6, Q7, Q10) par des questions
   utilisant predict_dossier / get_segment_scoring uniquement.
2. Envoie TOUTES les 10 questions au vrai agent Gemini (run_agent).
3. Stocke les vraies réponses dans agent_qa.json.
4. Génère chatbot_qa_report.pdf via generate_chatbot_report.py.
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── Ajouter src/ au path pour importer l'agent ───────────────────────────────
sys.path.insert(0, str(BASE_DIR))

# Importer l'agent réel
from src.api.services.agent_service import run_agent

# ── 10 questions — AUCUNE ne fait appel à query_history ──────────────────────
# Q2, Q6, Q7, Q10 remplacées par predict_dossier / get_segment_scoring

QUESTIONS = [
    {
        "key": "Q1",
        "language": "fr",
        "tools_hint": ["predict_dossier"],
        "question": (
            "Mon client Corporate a 45 000 MAD impayés depuis 8 mois, "
            "3 incidents de paiement, CDI privé, marié 2 enfants, revenu 80 000 MAD, "
            "8 échéances impayées. Quelle procédure recommandes-tu et quelle est "
            "la probabilité de récupérer la créance ?"
        ),
    },
    {
        "key": "Q2",
        "language": "fr",
        "tools_hint": ["predict_dossier"],
        "question": (
            "J'ai deux dossiers similaires en montant (30 000 MAD, 120 jours) : "
            "le premier est un client Retail CDI privé locataire avec 2 incidents, "
            "le deuxième est un client Professionnel indépendant propriétaire avec 1 incident. "
            "Analyse les deux profils avec le modèle ML et dis-moi lequel a la meilleure "
            "probabilité de recouvrement et quelle procédure choisir pour chacun."
        ),
    },
    {
        "key": "Q3",
        "language": "fr",
        "tools_hint": ["predict_dossier", "get_segment_scoring"],
        "question": (
            "Dossier urgent : client Professionnel, travailleur indépendant, locataire, "
            "120 000 MAD impayés depuis 200 jours, 10 échéances, revenu 45 000 MAD, "
            "1 incident. Durée de procédure estimée et première action à mener ?"
        ),
    },
    {
        "key": "Q4",
        "language": "fr",
        "tools_hint": ["predict_dossier"],
        "question": (
            "Client Retail, salaire domicilié, fonctionnaire État, propriétaire, "
            "6 échéances impayées depuis 300 jours, 15 000 MAD, revenu 12 000 MAD, "
            "2 incidents. Je négocie encore ou judiciaire direct ?"
        ),
    },
    {
        "key": "Q5",
        "language": "fr",
        "tools_hint": ["get_segment_scoring"],
        "question": (
            "Donne-moi le scoring complet des avocats et huissiers pour le segment Retail "
            "sur les dossiers judiciaires — performent-ils bien ?"
        ),
    },
    {
        "key": "Q6",
        "language": "fr",
        "tools_hint": ["predict_dossier"],
        "question": (
            "Client Retail, sans emploi, locataire, taux d'endettement estimé à 85%, "
            "22 000 MAD impayés depuis 180 jours, 7 échéances, 0 incident, revenu 0. "
            "Est-ce que le modèle recommande une procédure judiciaire ou amiable ? "
            "Et quelle est la probabilité réelle de recouvrement sur ce profil ?"
        ),
    },
    {
        "key": "Q7",
        "language": "fr",
        "tools_hint": ["predict_dossier", "get_segment_scoring"],
        "question": (
            "Profil Corporate à risque : client CDI privé, marié 3 enfants, "
            "propriétaire, revenu 120 000 MAD, 95 000 MAD impayés depuis 250 jours, "
            "12 échéances, 4 incidents. Donne-moi la prédiction complète du modèle "
            "ET le scoring avocat/huissier Corporate pour savoir qui mandater en priorité."
        ),
    },
    {
        "key": "Q8",
        "language": "darija",
        "tools_hint": ["predict_dossier", "get_segment_scoring"],
        "question": (
            "3andi client Corporate, 60,000 MAD ma3ando ypay depuis 10 chhor, "
            "5 incidents, CDI privé, revenu 90,000 MAD. "
            "Chno t9der t9oul liya — judiciaire wella amiable ? "
            "Wach kayn chance nrjou l flous ?"
        ),
    },
    {
        "key": "Q9",
        "language": "darija",
        "tools_hint": ["predict_dossier"],
        "question": (
            "3andi client Retail, 200 jour ma3ando ypay, 18,000 MAD, "
            "fonctionnaire Etat, mra o joj wlad, domiciliation salaire, 1 incident. "
            "Wakha tatla3 l procédure judiciaire, kifach ndir — "
            "saisie wella injonction de payer ?"
        ),
    },
    {
        "key": "Q10",
        "language": "fr",
        "tools_hint": ["get_segment_scoring", "predict_dossier"],
        "question": (
            "Compare le scoring avocat et huissier pour les segments Retail et Corporate. "
            "Ensuite, simule un dossier Corporate typique (revenu 75 000 MAD, "
            "55 000 MAD impayés, 210 jours, 3 incidents, CDI privé) et dis-moi "
            "quel acteur — avocat ou huissier — a le meilleur score pour traiter ce dossier, "
            "et quelle action le modèle recommande en premier."
        ),
    },
]

# ── Appel réel à Gemini ───────────────────────────────────────────────────────

RETRY_DELAY = 65      # secondes entre chaque appel (free tier = 15 RPM max)
MAX_RETRIES = 3       # nombre de tentatives par question

def is_error_answer(text: str) -> bool:
    """Retourne True si la réponse est une erreur (pas une vraie réponse Gemini)."""
    return text.strip().startswith("Erreur de l'agent")


def is_real_gemini_answer(item: dict) -> bool:
    """Retourne True uniquement si la réponse vient du vrai agent Gemini."""
    return item.get("source") == "gemini" and not is_error_answer(item.get("answer", ""))


def call_agent_with_retry(question: str, session_id: str) -> str:
    """Appelle l'agent avec retry et délai exponentiel en cas de rate limit."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            answer = run_agent(question, session_id=session_id)
            if is_error_answer(answer):
                raise Exception(answer)
            return answer
        except Exception as e:
            err = str(e)
            # Extraire le délai suggéré par Google si disponible
            m = re.search(r"retry in (\d+)", err)
            suggested = int(m.group(1)) if m else RETRY_DELAY
            wait = max(suggested + 5, RETRY_DELAY)
            if attempt < MAX_RETRIES:
                print(f"       ⏳ Rate limit (tentative {attempt}/{MAX_RETRIES}) — attente {wait}s...")
                time.sleep(wait)
            else:
                return f"Erreur de l'agent : {err}"
    return f"Erreur de l'agent : max retries dépassé"


# ── Charger les Q&A existantes pour ne rejouer que celles en erreur ──────────
qa_file = BASE_DIR / "agent_qa.json"
existing: dict = {}
if qa_file.exists():
    try:
        for item in json.load(open(qa_file, encoding="utf-8")):
            existing[item["key"]] = item
    except Exception:
        pass

print(f"\n{'='*60}")
print("Envoi des questions au vrai agent Gemini...")
print(f"Délai entre appels : {RETRY_DELAY}s  |  Max retries : {MAX_RETRIES}")
print(f"{'='*60}\n")

qa_results = []

for i, q in enumerate(QUESTIONS, 1):
    session_id = f"pdf-qa-{q['key'].lower()}"
    lang_label = "🇲🇦 DARIJA" if q["language"] == "darija" else "🇫🇷 FR"

    # Vérifier si on a déjà une vraie réponse Gemini (source=gemini)
    prev = existing.get(q["key"])
    if prev and is_real_gemini_answer(prev):
        print(f"[{i:02d}/10] {lang_label} | {q['key']} — ♻️  réponse Gemini réelle conservée")
        qa_results.append(prev)
        continue

    print(f"[{i:02d}/10] {lang_label} | {q['key']} — envoi à Gemini...")
    print(f"       Question : {q['question'][:80]}...")

    answer = call_agent_with_retry(q["question"], session_id)

    if is_error_answer(answer):
        print(f"       ❌ Échec après {MAX_RETRIES} tentatives\n")
    else:
        print(f"       ✅ Réponse réelle reçue ({len(answer)} chars)\n")

    qa_results.append({
        "key": q["key"],
        "question": q["question"],
        "answer": answer,
        "tools_used": q["tools_hint"],
        "language": q["language"],
        "source": "gemini" if not is_error_answer(answer) else "error",
    })

    # Pause entre appels (sauf dernier)
    if i < len(QUESTIONS):
        print(f"       ⏸  Pause {RETRY_DELAY}s (rate limit free tier)...")
        time.sleep(RETRY_DELAY)

# ── Sauvegarde dans agent_qa.json ─────────────────────────────────────────────
ok_count = sum(1 for r in qa_results if not is_error_answer(r["answer"]))
err_count = len(qa_results) - ok_count

with open(qa_file, "w", encoding="utf-8") as f:
    json.dump(qa_results, f, ensure_ascii=False, indent=2)

print(f"\n✅ agent_qa.json mis à jour : {ok_count} réponses OK, {err_count} erreurs")

# ── Génération du PDF ─────────────────────────────────────────────────────────
print("\nGénération du PDF chatbot en cours...")
result = subprocess.run(
    [sys.executable, str(BASE_DIR / "generate_chatbot_report.py")],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("ERREUR génération PDF :", result.stderr)
    sys.exit(1)

print("=" * 60)
print("DONE — chatbot_qa_report.pdf généré avec les vraies réponses Gemini !")
print("=" * 60)
