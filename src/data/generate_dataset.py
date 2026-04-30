import pandas as pd
import numpy as np
from faker import Faker
import logging
import uuid
import os
from pathlib import Path
from datetime import datetime
import warnings

# Détermination du chemin absolu de la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "generation_dataset.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class SyntheticDebtCollectionGenerator:
    """
    Générateur de dataset synthétique étendu pour les dossiers de recouvrement.
    Génère toutes les features nécessaires aux 4 modèles ML :
      1. Classification (statut_final)
      2. Régression (delai_final_jours)
      3. Next Best Action (next_best_action)
      4. Clustering (KMeans)
    + les métriques de performance historique (scoring avocat/huissier).
    """

    def __init__(self, num_records: int = 10000, random_seed: int = 42):
        self.num_records = num_records
        self.seed = random_seed
        self.faker = Faker('fr_FR')
        Faker.seed(self.seed)
        np.random.seed(self.seed)
        logger.info(f"Initialisation du générateur avec {self.num_records} enregistrements (Seed: {self.seed}).")

    def _generate_base_features(self) -> pd.DataFrame:
        """Identifiants et caractéristiques client."""
        logger.info("Génération des features de base (identifiants, client)...")
        df = pd.DataFrame()

        # Identifiants
        df['feature_id'] = [str(uuid.uuid4()) for _ in range(self.num_records)]
        df['dossier_id'] = [f"DOS-{i:06d}" for i in range(1, self.num_records + 1)]
        df['procedure_id'] = [f"PROC-{self.faker.unique.random_int(min=1000, max=999999)}" for _ in range(self.num_records)]

        # Client
        segments = ['Retail', 'Professionnel', 'Corporate']
        df['client_segment'] = np.random.choice(segments, self.num_records, p=[0.75, 0.20, 0.05])
        df['historique_incidents'] = np.random.choice([0, 1, 2, 3, 4, 5], self.num_records, p=[0.5, 0.25, 0.15, 0.05, 0.03, 0.02])

        # Dates
        dates_ouverture = [self.faker.date_between(start_date='-3y', end_date='-1m') for _ in range(self.num_records)]
        df['date_ouverture'] = pd.to_datetime(dates_ouverture)

        return df

    def _generate_financial_and_debt_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Variables financières et détails de la dette."""
        logger.info("Génération des variables financières et dette...")

        revenus, montants = [], []
        for segment in df['client_segment']:
            if segment == 'Retail':
                rev, montant = np.random.lognormal(10.5, 0.5), np.random.lognormal(6.5, 1.0)
            elif segment == 'Professionnel':
                rev, montant = np.random.lognormal(11.5, 0.8), np.random.lognormal(8.0, 1.2)
            else:
                rev, montant = np.random.lognormal(13.5, 1.0), np.random.lognormal(10.0, 1.5)
            revenus.append(max(500.0, rev))
            montants.append(max(50.0, montant))

        df['revenu_estime'] = np.round(revenus, 2)
        df['montant_impaye'] = np.round(montants, 2)
        df['anciennete_impaye_jours'] = np.random.randint(30, 1000, self.num_records)
        df['nombre_echeances_impayees'] = np.random.randint(1, 24, self.num_records)

        return df

    def _generate_procedure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caractéristiques de la procédure et identifiants des acteurs."""
        logger.info("Génération des caractéristiques de la procédure...")

        # Score de risque
        ratio_endettement = df['montant_impaye'] / df['revenu_estime']
        base_score = 30 + 100 * (1 - np.exp(-ratio_endettement * 2))
        df['score_risque'] = np.clip(np.round(base_score + np.random.normal(0, 12, self.num_records)), 1, 100).astype(int)

        # Type de procédure
        prob_judiciaire = 1 / (1 + np.exp(-(df['score_risque'] - 75) / 10))
        df['type_procedure'] = np.where(np.random.uniform(0, 1, self.num_records) < prob_judiciaire, 'Judiciaire', 'Amiable')

        # Identifiants acteurs
        df['tribunal_id'] = [f"TRIB-{np.random.randint(1, 20):02d}" if p == 'Judiciaire' else 'NONE' for p in df['type_procedure']]
        df['avocat_id'] = [f"AVO-{np.random.randint(1, 50):03d}" if p == 'Judiciaire' else 'NONE' for p in df['type_procedure']]
        df['huissier_id'] = [f"HUI-{np.random.randint(1, 30):03d}" for _ in range(self.num_records)]

        # Événements
        df['nombre_evenements'] = np.random.randint(1, 15, self.num_records)
        df['nombre_retards'] = np.random.randint(0, 5, self.num_records)
        df['derniere_action_age_jours'] = np.random.randint(1, 60, self.num_records)

        return df

    def _generate_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Métriques de performance historique des acteurs.
        En production, ces valeurs seront calculées depuis PostgreSQL.
        Ici, on les simule pour l'entraînement.
        """
        logger.info("Génération des métriques de performance historique...")

        df['acteur_taux_succes'] = np.round(np.random.beta(a=4, b=2, size=self.num_records), 3)
        df['acteur_delai_moyen'] = np.round(np.random.normal(loc=120, scale=30, size=self.num_records).clip(min=10), 1)
        df['tribunal_delai_moyen'] = np.where(
            df['type_procedure'] == 'Judiciaire',
            np.round(np.random.normal(loc=180, scale=45, size=self.num_records).clip(min=30), 1),
            0.0
        )
        df['procedure_taux_succes'] = (
            np.where(df['type_procedure'] == 'Judiciaire', 0.65, 0.85)
            + np.random.normal(0, 0.05, self.num_records)
        )
        df['procedure_taux_succes'] = df['procedure_taux_succes'].clip(0, 1).round(3)

        return df

    def _generate_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les 3 labels d'entraînement :
          - statut_final (Classification)
          - delai_final_jours (Régression)
          - next_best_action (Classification multi-classe)
        """
        logger.info("Génération des variables cibles (statut, délai, next action)...")

        # --- 1. Statut Final ---
        logit_p = (
            1.0 - 0.03 * df['score_risque']
            + 2.5 * df['acteur_taux_succes']
            - 0.1 * df['historique_incidents']
            - 0.001 * df['anciennete_impaye_jours']
            + np.where(df['type_procedure'] == 'Amiable', 0.8, -0.5)
            + np.random.logistic(0, 1.5, self.num_records)
        )
        prob_success = 1 / (1 + np.exp(-logit_p))

        rand_draw = np.random.uniform(0, 1, self.num_records)
        df['statut_final'] = np.select(
            [rand_draw < prob_success * 0.75, rand_draw > (1 - (1 - prob_success) * 0.65)],
            ['Recouvré', 'Échec'], default='En cours'
        )

        # --- 2. Délai Final ---
        delais_jours = np.random.randint(15, 400, self.num_records)
        df['delai_final_jours'] = delais_jours
        df['date_mise_a_jour'] = df['date_ouverture'] + pd.to_timedelta(delais_jours, unit='D')

        # --- 3. Montant Recouvré ---
        df['montant_recouvre_final'] = np.where(
            df['statut_final'] == 'Recouvré', df['montant_impaye'],
            np.where(df['statut_final'] == 'En cours',
                     df['montant_impaye'] * np.random.uniform(0, 0.5, self.num_records), 0.0)
        ).round(2)

        # --- 4. Next Best Action ---
        # Dérivé des données : quelle action a mené au meilleur résultat dans cette situation ?
        conditions = [
            # Cas facile : risque faible + procédure amiable → Relance amiable
            (df['score_risque'] < 40) & (df['type_procedure'] == 'Amiable'),
            # Cas modéré : risque moyen + retards → Mise en demeure
            (df['score_risque'].between(40, 70)) & (df['nombre_retards'] >= 1),
            # Cas difficile : risque élevé + procédure judiciaire → Action judiciaire
            (df['score_risque'] > 70) & (df['type_procedure'] == 'Judiciaire'),
            # Cas bloqué : ancienneté très longue → Négociation / Plan de paiement
            (df['anciennete_impaye_jours'] > 300) | (df['nombre_echeances_impayees'] > 10),
        ]
        choices = ['Relance amiable', 'Mise en demeure', 'Action judiciaire', 'Négociation']
        df['next_best_action'] = np.select(conditions, choices, default='Relance amiable')

        return df

    def generate(self) -> pd.DataFrame:
        """Orchestrateur principal."""
        start_time = datetime.now()

        df = self._generate_base_features()
        df = self._generate_financial_and_debt_features(df)
        df = self._generate_procedure_features(df)
        df = self._generate_performance_metrics(df)
        df = self._generate_target_variables(df)

        # Sauvegarde
        path_dir = BASE_DIR / 'data'
        path_dir.mkdir(parents=True, exist_ok=True)
        filepath = path_dir / 'recouvrement_dataset.csv'
        df.to_csv(filepath, index=False, encoding='utf-8')

        logger.info(f"Génération complétée en {(datetime.now() - start_time).total_seconds():.2f}s.")
        logger.info(f"Dataset sauvegardé : {filepath} ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df


if __name__ == "__main__":
    generator = SyntheticDebtCollectionGenerator(num_records=10000)
    df_result = generator.generate()
    print(f"\nColonnes générées ({len(df_result.columns)}) :", df_result.columns.tolist())
    print("\n--- Distribution Statut Final ---")
    print(df_result['statut_final'].value_counts())
    print("\n--- Distribution Next Best Action ---")
    print(df_result['next_best_action'].value_counts())
