import pandas as pd
import numpy as np
from faker import Faker
import logging
import uuid
import os
from pathlib import Path
from datetime import datetime
import warnings

# Configuration d'un système de logging robuste
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../../logs/generation_dataset.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

class SyntheticDebtCollectionGenerator:
    """
    Générateur de dataset synthétique pour les dossiers de recouvrement.
    Architecture orientée objet avec gestion des erreurs, prévention du data leakage et validation stricte.
    """
    
    def __init__(self, num_records: int = 10000, random_seed: int = 42):
        self.num_records = num_records
        self.seed = random_seed
        self.faker = Faker('fr_FR')
        # Fixer les seeds pour garantir la reproductibilité de la génération
        Faker.seed(self.seed)
        np.random.seed(self.seed)
        logger.info(f"Initialisation du générateur avec {self.num_records} enregistrements (Seed: {self.seed}).")

    def _generate_base_features(self) -> pd.DataFrame:
        """Génère les identifiants, variables catégorielles et temporelles (String, DateTime)."""
        logger.info("Génération des identifiants, variables catégorielles et temporelles...")
        
        df = pd.DataFrame()
        # Identifiants (Strings)
        df['feature_id'] = [str(uuid.uuid4()) for _ in range(self.num_records)]
        df['dossier_id'] = [f"DOS-{i:06d}" for i in range(1, self.num_records + 1)]
        df['procedure_id'] = [f"PROC-{self.faker.unique.random_int(min=1000, max=999999)}" for _ in range(self.num_records)]
        
        # Segments clients avec des probabilités réalistes
        segments = ['Retail', 'Professionnel', 'Corporate']
        df['client_segment'] = np.random.choice(segments, self.num_records, p=[0.75, 0.20, 0.05])

        # Dates (DateTime) - Assure la cohérence temporelle
        dates_ouverture = [self.faker.date_between(start_date='-3y', end_date='-1m') for _ in range(self.num_records)]
        df['date_ouverture'] = pd.to_datetime(dates_ouverture)
        
        # Date de statut avec délai logique
        delais_jours = np.random.randint(5, 400, self.num_records)
        df['date_mise_a_jour'] = df['date_ouverture'] + pd.to_timedelta(delais_jours, unit='D')
        # S'assurer que la date de mise à jour ne dépasse pas aujourd'hui
        df['date_mise_a_jour'] = df['date_mise_a_jour'].clip(upper=pd.Timestamp.today())
        
        return df

    def _generate_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les variables financières (Float) adaptées aux segments clients."""
        logger.info("Génération des variables financières (Float)...")
        
        revenus = []
        montants = []
        for segment in df['client_segment']:
            # Le revenu et l'impayé sont corrélés au segment client avec une distribution log-normale
            if segment == 'Retail':
                rev = np.random.lognormal(mean=10.5, sigma=0.5) 
                montant = np.random.lognormal(mean=6.5, sigma=1.0)
            elif segment == 'Professionnel':
                rev = np.random.lognormal(mean=11.5, sigma=0.8)
                montant = np.random.lognormal(mean=8.0, sigma=1.2)
            else: # Corporate
                rev = np.random.lognormal(mean=13.5, sigma=1.0)
                montant = np.random.lognormal(mean=10.0, sigma=1.5)
                
            revenus.append(max(500.0, rev))
            montants.append(max(50.0, montant))
            
        df['revenu_estime'] = np.round(revenus, 2)
        df['montant_impaye'] = np.round(montants, 2)
        return df

    def _generate_risk_and_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère le risque et les caractéristiques de la procédure (Integer, Float, String)."""
        logger.info("Génération du risque et des caractéristiques de procédure...")
        
        # Score de risque : Combinaison non-linéaire du taux d'endettement avec ajout de bruit
        ratio_endettement = df['montant_impaye'] / df['revenu_estime']
        base_score = 30 + 100 * (1 - np.exp(-ratio_endettement * 2))
        bruit = np.random.normal(loc=0, scale=12, size=self.num_records)
        scores = base_score + bruit
        
        df['score_risque'] = np.clip(np.round(scores), 1, 100).astype(int)
        
        # Type de procédure : un score élevé augmente les chances de procédure Judiciaire
        prob_judiciaire = 1 / (1 + np.exp(-(df['score_risque'] - 75) / 10)) 
        rand_vals = np.random.uniform(0, 1, self.num_records)
        df['type_procedure'] = np.where(rand_vals < prob_judiciaire, 'Judiciaire', 'Amiable')
        
        # Taux de succès historique de l'acteur assigné (Float: 0.0 - 1.0)
        df['acteur_taux_succes'] = np.round(np.random.beta(a=4, b=2, size=self.num_records), 3)
        return df

    def _generate_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère la variable cible 'statut_final' avec prévention du Data Leakage.
        Le label est probabiliste et stochastique, empêchant le modèle de trouver une formule parfaite.
        """
        logger.info("Génération de la variable cible 'statut_final' (Prévention du Data Leakage)...")
        
        # Variable latente (combinaison linéaire des features importantes)
        logit_p = (
            1.0 
            - 0.03 * df['score_risque'] 
            + 2.5 * df['acteur_taux_succes']
            - 0.00005 * df['montant_impaye']
            + np.where(df['type_procedure'] == 'Amiable', 0.8, -0.5)
        )
        
        # Bruit très important pour éviter un R² parfait et forcer le modèle ML à généraliser
        noise = np.random.logistic(loc=0, scale=1.5, size=self.num_records)
        logit_p += noise
        
        prob_success = 1 / (1 + np.exp(-logit_p))
        
        # Tirage stochastique pour définir la catégorie
        rand_draw = np.random.uniform(0, 1, self.num_records)
        conditions = [
            rand_draw < prob_success * 0.75, 
            rand_draw > (1 - (1 - prob_success) * 0.65), 
        ]
        choices = ['Recouvré', 'Échec']
        df['statut_final'] = np.select(conditions, choices, default='En cours')
        return df

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Valide l'intégrité métier et les types de données avant l'exportation."""
        logger.info("Validation de l'intégrité métier et des types de données...")
        try:
            # Vérification des types imposés
            assert pd.api.types.is_datetime64_any_dtype(df['date_ouverture']), "Type DateTime manquant ou incorrect."
            assert pd.api.types.is_float_dtype(df['revenu_estime']), "Type Float manquant ou incorrect."
            assert pd.api.types.is_integer_dtype(df['score_risque']), "Type Integer manquant ou incorrect."
            
            # Vérifications logiques métier
            assert (df['revenu_estime'] > 0).all(), "Données aberrantes : Revenus négatifs détectés."
            assert (df['montant_impaye'] >= 0).all(), "Données aberrantes : Montants impayés négatifs détectés."
            assert df['score_risque'].between(1, 100).all(), "Données aberrantes : Score de risque hors bornes [1, 100]."
            assert df['acteur_taux_succes'].between(0, 1).all(), "Données aberrantes : Taux de succès acteur invalide [0, 1]."
            assert not df.isnull().any().any(), "Validation échouée : Présence de valeurs manquantes (NaN/NaT)."
            
            logger.info("Toutes les règles de validation ont été respectées.")
            return True
        except AssertionError as e:
            logger.error(f"Échec de validation des données : {e}")
            raise

    def save_securely(self, df: pd.DataFrame, output_dir: str = 'data', filename: str = 'recouvrement_dataset.csv'):
        """Sauvegarde sécurisée : crée le dossier si inexistant et prévient la corruption des fichiers."""
        logger.info(f"Préparation de la sauvegarde sécurisée dans {output_dir}/{filename}...")
        try:
            # Sécurité 1 : Gestion sécurisée du répertoire (création sans erreur si existant)
            # Utilisation du chemin absolu racine du projet
            base_path = Path(__file__).resolve().parent.parent.parent
            path_dir = base_path / output_dir
            path_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = path_dir / filename
            temp_filepath = path_dir / f"{filename}.tmp"
            
            # Sécurité 2 : Sauvegarde atomique (via fichier temporaire)
            logger.info("Écriture dans un fichier temporaire pour éviter les corruptions en cas de crash...")
            df.to_csv(temp_filepath, index=False, encoding='utf-8')
            
            # Sécurité 3 : Gestion d'un éventuel écrasement
            if filepath.exists():
                backup_name = f"{filename.replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                backup_path = path_dir / backup_name
                os.rename(filepath, backup_path)
                logger.warning(f"Fichier existant détecté. Backup de l'ancien dataset créé : {backup_path}")
            
            # Déplacement final (quasi-atomique selon l'OS)
            os.rename(temp_filepath, filepath)
            logger.info(f"SUCCESS : Le dataset de {self.num_records} lignes a été généré et sauvegardé avec succès.")
            logger.info(f"Chemin absolu : {filepath.absolute()}")
            
        except Exception as e:
            logger.error(f"Erreur critique lors de la création/sauvegarde du fichier : {e}")
            # Nettoyage
            if 'temp_filepath' in locals() and temp_filepath.exists():
                os.remove(temp_filepath)
                logger.info("Nettoyage du fichier temporaire effectué suite à l'erreur.")
            raise

    def generate(self) -> pd.DataFrame:
        """Orchestrateur principal avec gestion des erreurs."""
        try:
            start_time = datetime.now()
            
            df = self._generate_base_features()
            df = self._generate_financial_features(df)
            df = self._generate_risk_and_process_features(df)
            df = self._generate_target_variable(df)
            
            self._validate_data(df)
            self.save_securely(df, output_dir='data', filename='recouvrement_dataset.csv')
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Génération complétée en {duration:.2f} secondes.")
            return df
            
        except Exception as e:
            logger.critical(f"Arrêt du processus de génération suite à une exception : {e}")
            raise

if __name__ == "__main__":
    # Exécution du script
    generator = SyntheticDebtCollectionGenerator(num_records=10000)
    df_result = generator.generate()
    
    print("\n--- APERÇU DES DONNÉES GÉNÉRÉES ---")
    print(df_result[['dossier_id', 'client_segment', 'score_risque', 'montant_impaye', 'type_procedure', 'statut_final']].head(10))
    print("\n--- DISTRIBUTION DU STATUT FINAL ---")
    print((df_result['statut_final'].value_counts(normalize=True).round(3) * 100).to_string() + " %")
