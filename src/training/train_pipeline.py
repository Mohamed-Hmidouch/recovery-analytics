import os
import logging
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

# -- Configuration des chemins et du logging --
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "training_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Initialise la session locale PySpark."""
    logger.info("Initialisation de la session Spark locale...")
    return SparkSession.builder \
        .appName("SmartRecovery-ML-Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def load_data(spark):
    """Charge le dataset généré par le script de data generation."""
    data_path = str(BASE_DIR / "data" / "recouvrement_dataset.csv")
    logger.info(f"Chargement des données depuis {data_path}...")
    
    if not os.path.exists(data_path):
        logger.error("Le fichier de données est introuvable. Avez-vous exécuté le script generate_dataset.py ?")
        raise FileNotFoundError(f"Fichier manquant : {data_path}")
        
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info(f"Dataset chargé avec succès : {df.count()} lignes, {len(df.columns)} colonnes.")
    return df

def feature_engineering(df):
    """Application des transformations métier."""
    logger.info("Début du Feature Engineering...")
    
    # 1. Calcul du delai_final_jours
    df = df.withColumn("delai_final_jours", datediff(col("date_mise_a_jour"), col("date_ouverture")))
    
    # 2. Calcul du Score d'Avocat (score métier composite)
    # Plus l'avocat a de succès et moins il met de temps, meilleur est son score
    df = df.withColumn(
        "score_avocat", 
        expr("(acteur_taux_succes * 100) - (delai_final_jours * 0.05)")
    )
    
    # Filtrer les éventuelles lignes avec délai négatif (incohérences temporelles)
    df = df.filter(col("delai_final_jours") >= 0)
    
    logger.info("Feature Engineering terminé.")
    return df

def build_pipelines():
    """Construit les Pipelines de pre-processing, classification et régression."""
    logger.info("Construction du pipeline ML PySpark...")
    
    # --- Prétraitement (partagé) ---
    categorical_cols = ["client_segment", "type_procedure"]
    indexers = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCols=[c+"_index" for c in categorical_cols], 
                              outputCols=[c+"_ohe" for c in categorical_cols])]
    
    # Label pour la classification
    label_indexer = StringIndexer(inputCol="statut_final", outputCol="label_class", handleInvalid="keep")
    
    # Features numériques
    numeric_cols = ["revenu_estime", "score_risque", "montant_impaye", "acteur_taux_succes", "score_avocat"]
    
    # Assemblage final des features
    assembler_inputs = [c+"_ohe" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_unscaled")
    
    # Standardisation
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    
    preprocessing_stages = indexers + encoders + [label_indexer, assembler, scaler]
    
    # --- Modèles ---
    rf_classifier = RandomForestClassifier(labelCol="label_class", featuresCol="features", numTrees=50, maxDepth=5, seed=42)
    rf_regressor = RandomForestRegressor(labelCol="delai_final_jours", featuresCol="features", numTrees=50, maxDepth=5, seed=42)
    
    # Pipelines distincts
    pipeline_class = Pipeline(stages=preprocessing_stages + [rf_classifier])
    pipeline_reg = Pipeline(stages=preprocessing_stages + [rf_regressor])
    
    return pipeline_class, pipeline_reg

def train_and_evaluate(df, pipeline_class, pipeline_reg):
    """Entraîne et évalue les modèles."""
    logger.info("Division des données en Train (80%) et Test (20%)...")
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # --- Modèle 1: Classification ---
    logger.info("Entraînement du modèle Multi-Target 1 : RandomForestClassifier (statut_final)...")
    model_class = pipeline_class.fit(train_data)
    predictions_class = model_class.transform(test_data)
    
    evaluator_class = MulticlassClassificationEvaluator(labelCol="label_class", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_class.evaluate(predictions_class)
    logger.info(f"[METRIQUE] Accuracy du Classifier : {accuracy:.4f}")
    
    # --- Modèle 2: Régression ---
    logger.info("Entraînement du modèle Multi-Target 2 : RandomForestRegressor (delai_final_jours)...")
    model_reg = pipeline_reg.fit(train_data)
    predictions_reg = model_reg.transform(test_data)
    
    evaluator_reg = RegressionEvaluator(labelCol="delai_final_jours", predictionCol="prediction", metricName="rmse")
    rmse = evaluator_reg.evaluate(predictions_reg)
    logger.info(f"[METRIQUE] RMSE du Regressor (en jours) : {rmse:.4f}")
    
    return model_class, model_reg

def save_models(model_class, model_reg):
    """Exporte les modèles entraînés avec un horodatage pour éviter les écrasements."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = BASE_DIR / "models"
    
    path_class = str(models_dir / f"rf_classifier_{timestamp}")
    path_reg = str(models_dir / f"rf_regressor_{timestamp}")
    
    logger.info(f"Sauvegarde du RandomForestClassifier dans {path_class}...")
    model_class.write().overwrite().save(path_class)
    
    logger.info(f"Sauvegarde du RandomForestRegressor dans {path_reg}...")
    model_reg.write().overwrite().save(path_reg)

def main():
    spark = None
    try:
        spark = create_spark_session()
        
        # Ingestion
        df = load_data(spark)
        
        # Engineering
        df_engineered = feature_engineering(df)
        
        # Pipelines
        pipeline_class, pipeline_reg = build_pipelines()
        
        # Entraînement
        model_class, model_reg = train_and_evaluate(df_engineered, pipeline_class, pipeline_reg)
        
        # Export
        save_models(model_class, model_reg)
        
        logger.info("Pipeline d'entraînement terminé avec succès. Tous les modèles sont sauvegardés.")
        
    except Exception as e:
        logger.error(f"Erreur fatale lors de l'exécution du pipeline: {str(e)}", exc_info=True)
        raise
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Session Spark arrêtée.")

if __name__ == "__main__":
    main()
