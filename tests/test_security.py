"""Tests de sécurité automatisés pour l'API SmartRecovery."""
import requests
import json
import sys

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
VALID_KEY = os.getenv("SECRET_API_KEY")

if not VALID_KEY:
    print("Erreur critique: SECRET_API_KEY est introuvable dans le .env.")
    sys.exit(1)

# Si l'hôte est 0.0.0.0, on utilise 127.0.0.1 pour le test local
HOST = "127.0.0.1" if API_HOST == "0.0.0.0" else API_HOST
BASE = f"http://{HOST}:{API_PORT}"
ENDPOINT = f"{BASE}/api/v1/predict/recouvrement"
HEALTH = f"{BASE}/api/v1/health"

VALID_PAYLOAD = {
    "client_segment": "Retail",
    "revenu_estime": 45000.0,
    "score_risque": 62,
    "montant_impaye": 8500.50,
    "type_procedure": "Amiable",
    "acteur_taux_succes": 0.72,
    "date_ouverture": "2024-03-15",
    "date_mise_a_jour": "2024-09-20"
}

results = []

def test(name, response, expected_code):
    status = "PASS" if response.status_code == expected_code else "FAIL"
    results.append((name, status, response.status_code, expected_code))
    print(f"[{status}] {name} — HTTP {response.status_code} (attendu: {expected_code})")
    if response.status_code != expected_code:
        print(f"       Body: {response.text[:200]}")

# TEST 1: Health sans auth
r = requests.get(HEALTH)
test("Health check (pas d'auth)", r, 200)

# TEST 2: Pas de clé API
r = requests.post(ENDPOINT, json=VALID_PAYLOAD)
test("Requete sans X-API-Key", r, 401)

# TEST 3: Mauvaise clé API
r = requests.post(ENDPOINT, json=VALID_PAYLOAD, headers={"X-API-Key": "FAKE-HACKER-KEY"})
test("Mauvaise API Key", r, 403)

# TEST 4: Bonne clé API + payload valide
r = requests.post(ENDPOINT, json=VALID_PAYLOAD, headers={"X-API-Key": VALID_KEY})
test("Prediction valide", r, 200)

# TEST 5: JSON Bomb (payload > 1 MB)
bomb = {"client_segment": "A" * 2_000_000}
r = requests.post(ENDPOINT, json=bomb, headers={"X-API-Key": VALID_KEY})
test("JSON Bomb (>1MB)", r, 413)

# TEST 6: Rate Limit (6eme requete dans la meme minute)
for i in range(6):
    r = requests.post(ENDPOINT, json=VALID_PAYLOAD, headers={"X-API-Key": VALID_KEY})
test("Rate Limit (6eme requete)", r, 429)

print("\n--- RÉSUMÉ ---")
passed = sum(1 for _, s, _, _ in results if s == "PASS")
print(f"{passed}/{len(results)} tests passés.")
sys.exit(0 if passed == len(results) else 1)
