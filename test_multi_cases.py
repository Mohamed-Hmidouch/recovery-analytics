import asyncio
from fastapi.testclient import TestClient
from src.api.main import app
import os
import uuid
from datetime import date, timedelta

# Create test client
client = TestClient(app)

# We need a valid API key. The API uses DEPENDS(verify_api_key) which looks at X-API-Key.
# Let's check the .env file or just set it in the environment.
API_KEY = os.getenv("API_KEY", "smart_recovery_secret_key_2024")
headers = {"X-API-Key": API_KEY}

def test_dossier_favorable():
    print("\n--- Test Case 1: Dossier Favorable (Retail, Low Risk) ---")
    payload = {
        "dossier_id": str(uuid.uuid4()),
        "procedure_id": str(uuid.uuid4()),
        "client_segment": "Retail",
        "revenu_estime": 8000.0,
        "historique_incidents": 0,
        "montant_impaye": 500.0,
        "anciennete_impaye_jours": 30,
        "nombre_echeances_impayees": 1,
        "date_ouverture": (date.today() - timedelta(days=30)).isoformat(),
        "date_mise_a_jour": date.today().isoformat()
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/predict/recouvrement", json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("Status:", response.status_code)
            print("Meilleure Procédure (Next Best Action):", data.get("meilleure_procedure"))
            print("Taux de Succès:", data.get("taux_de_succes"))
            print("Prochaine Action Recommandée:", data.get("prochaine_action_recommandee"))
            print("Statut Final Prédit:", data.get("statut_final_predit"))
            print("Délai Estimé:", data.get("delai_estime_jours"))
        else:
            print("Failed:", response.status_code, response.text)

def test_dossier_critique():
    print("\n--- Test Case 2: Dossier Critique (Corporate, High Risk) ---")
    payload = {
        "dossier_id": str(uuid.uuid4()),
        "procedure_id": str(uuid.uuid4()),
        "client_segment": "Corporate",
        "revenu_estime": 1000.0,
        "historique_incidents": 5,
        "montant_impaye": 25000.0,
        "anciennete_impaye_jours": 180,
        "nombre_echeances_impayees": 6,
        "date_ouverture": (date.today() - timedelta(days=180)).isoformat(),
        "date_mise_a_jour": date.today().isoformat()
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/predict/recouvrement", json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("Status:", response.status_code)
            print("Meilleure Procédure (Next Best Action):", data.get("meilleure_procedure"))
            print("Taux de Succès:", data.get("taux_de_succes"))
            print("Prochaine Action Recommandée:", data.get("prochaine_action_recommandee"))
            print("Statut Final Prédit:", data.get("statut_final_predit"))
            print("Délai Estimé:", data.get("delai_estime_jours"))
        else:
            print("Failed:", response.status_code, response.text)

if __name__ == "__main__":
    test_dossier_favorable()
    test_dossier_critique()
    print("\nTesting complete.")
