"""
Pipeline d'entraînement Spark ML — 4 Modèles distincts.

  Modèle 1 : RandomForestClassifier  → Probabilité de recouvrement (statut_final)
  Modèle 2 : RandomForestRegressor   → Prédiction durée procédure (delai_final_jours)
  Modèle 3 : RandomForestClassifier  → Next Best Action (next_best_action)
  Modèle 4 : KMeans                  → Segmentation dossiers (clustering)
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator,
)

# -- Configuration --
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# -- Features partagées entre tous les modèles --
CATEGORICAL_COLS = ["client_segment", "type_procedure"]
NUMERIC_COLS = [
    "revenu_estime", "score_risque", "montant_impaye", "acteur_taux_succes",
    "score_avocat", "historique_incidents", "anciennete_impaye_jours",
    "nombre_echeances_impayees", "nombre_evenements", "nombre_retards",
    "derniere_action_age_jours", "acteur_delai_moyen", "tribunal_delai_moyen",
    "procedure_taux_succes",
]


def create_spark_session():
    return (
        SparkSession.builder
        .appName("SmartRecovery-ML-Pipeline")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_data(spark):
    data_path = str(BASE_DIR / "data" / "recouvrement_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier manquant : {data_path}")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info(f"Dataset chargé : {df.count()} lignes, {len(df.columns)} colonnes.")
    return df


def feature_engineering(df):
    """Ajoute les features calculées métier."""
    df = df.withColumn(
        "score_avocat",
        expr("(acteur_taux_succes * 100) - (delai_final_jours * 0.05)"),
    )
    return df


def _build_preprocessing_stages():
    """Construit les étapes de preprocessing partagées."""
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]
    encoders = [
        OneHotEncoder(
            inputCols=[c + "_index" for c in CATEGORICAL_COLS],
            outputCols=[c + "_ohe" for c in CATEGORICAL_COLS],
        )
    ]
    assembler = VectorAssembler(
        inputCols=[c + "_ohe" for c in CATEGORICAL_COLS] + NUMERIC_COLS,
        outputCol="features_unscaled",
    )
    scaler = StandardScaler(
        inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True
    )
    return indexers + encoders + [assembler, scaler]


def build_all_pipelines():
    """Construit les 4 pipelines ML."""
    preprocessing = _build_preprocessing_stages()

    # Label indexers
    label_statut = StringIndexer(inputCol="statut_final", outputCol="label_statut", handleInvalid="keep")
    label_action = StringIndexer(inputCol="next_best_action", outputCol="label_action", handleInvalid="keep")

    # Modèle 1 : Classification — Probabilité de recouvrement
    clf_statut = RandomForestClassifier(
        labelCol="label_statut", featuresCol="features", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_statut = Pipeline(stages=preprocessing + [label_statut, clf_statut])

    # Modèle 2 : Régression — Prédiction durée procédure
    reg_delai = RandomForestRegressor(
        labelCol="delai_final_jours", featuresCol="features", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_delai = Pipeline(stages=preprocessing + [label_statut, reg_delai])

    # Modèle 3 : Classification — Next Best Action
    clf_action = RandomForestClassifier(
        labelCol="label_action", featuresCol="features", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_action = Pipeline(stages=preprocessing + [label_statut, label_action, clf_action])

    # Modèle 4 : Clustering — Segmentation
    kmeans = KMeans().setK(4).setSeed(42).setFeaturesCol("features").setPredictionCol("cluster_id")
    pipeline_cluster = Pipeline(stages=preprocessing + [label_statut, kmeans])

    return pipeline_statut, pipeline_delai, pipeline_action, pipeline_cluster


def train_and_evaluate(df, p_statut, p_delai, p_action, p_cluster):
    """Entraîne et évalue les 4 modèles."""
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # 1. Classification — Statut Final
    logger.info("══ Modèle 1/4 : Classification (Probabilité de recouvrement) ══")
    m_statut = p_statut.fit(train_data)
    acc = MulticlassClassificationEvaluator(
        labelCol="label_statut", predictionCol="prediction", metricName="accuracy"
    ).evaluate(m_statut.transform(test_data))
    logger.info(f"   → Accuracy : {acc:.4f}")

    # 2. Régression — Délai
    logger.info("══ Modèle 2/4 : Régression (Prédiction durée procédure) ══")
    m_delai = p_delai.fit(train_data)
    rmse = RegressionEvaluator(
        labelCol="delai_final_jours", predictionCol="prediction", metricName="rmse"
    ).evaluate(m_delai.transform(test_data))
    logger.info(f"   → RMSE : {rmse:.4f} jours")

    # 3. Classification — Next Best Action
    logger.info("══ Modèle 3/4 : Classification (Next Best Action) ══")
    m_action = p_action.fit(train_data)
    acc_action = MulticlassClassificationEvaluator(
        labelCol="label_action", predictionCol="prediction", metricName="accuracy"
    ).evaluate(m_action.transform(test_data))
    logger.info(f"   → Accuracy : {acc_action:.4f}")

    # 4. Clustering — KMeans (non-supervisé, tout le dataset)
    logger.info("══ Modèle 4/4 : Clustering (Segmentation KMeans) ══")
    m_cluster = p_cluster.fit(df)
    silhouette = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster_id"
    ).evaluate(m_cluster.transform(df))
    logger.info(f"   → Silhouette : {silhouette:.4f}")

    return m_statut, m_delai, m_action, m_cluster


def save_models(m_statut, m_delai, m_action, m_cluster):
    """Sauvegarde les 4 modèles avec horodatage."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = BASE_DIR / "models"

    paths = {
        "rf_classifier": m_statut,
        "rf_regressor": m_delai,
        "rf_next_action": m_action,
        "kmeans_cluster": m_cluster,
    }
    for name, model in paths.items():
        path = str(models_dir / f"{name}_{timestamp}")
        model.write().overwrite().save(path)
        logger.info(f"   Modèle sauvegardé : {path}")


def main():
    spark = create_spark_session()
    try:
        df = load_data(spark)
        df = feature_engineering(df)
        p_statut, p_delai, p_action, p_cluster = build_all_pipelines()
        m_statut, m_delai, m_action, m_cluster = train_and_evaluate(
            df, p_statut, p_delai, p_action, p_cluster
        )
        save_models(m_statut, m_delai, m_action, m_cluster)
        logger.info("════ Entraînement terminé. 4 modèles sauvegardés avec succès. ════")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
