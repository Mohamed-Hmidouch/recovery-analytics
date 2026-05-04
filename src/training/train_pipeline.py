"""
Pipeline d'entraînement Spark ML — 5 Modèles distincts.

  Modèle 1 : RandomForestClassifier  → Prédiction type procédure (Amiable/Judiciaire) - NOUVEAU
  Modèle 2 : RandomForestClassifier  → Probabilité de recouvrement (statut_final)
  Modèle 3 : RandomForestRegressor   → Prédiction durée procédure (delai_final_jours)
  Modèle 4 : RandomForestClassifier  → Next Best Action (next_best_action)
  Modèle 5 : KMeans                  → Segmentation dossiers (clustering)
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

# -- Features de Base (Avant décision de la procédure) --
BASE_CATEGORICAL_COLS = ["client_segment"]
BASE_NUMERIC_COLS = [
    "revenu_estime", "score_risque", "montant_impaye", 
    "historique_incidents", "anciennete_impaye_jours", "nombre_echeances_impayees"
]

# -- Features Complètes (Inclus le type de procédure prédit et l'historique) --
FULL_CATEGORICAL_COLS = ["client_segment", "type_procedure"]
FULL_NUMERIC_COLS = [
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


def _build_preprocessing_stages(categorical_cols, numeric_cols, suffix):
    """Construit les étapes de preprocessing dynamiques."""
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + f"_index_{suffix}", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(
            inputCols=[c + f"_index_{suffix}" for c in categorical_cols],
            outputCols=[c + f"_ohe_{suffix}" for c in categorical_cols],
        )
    ]
    assembler = VectorAssembler(
        inputCols=[c + f"_ohe_{suffix}" for c in categorical_cols] + numeric_cols,
        outputCol=f"features_unscaled_{suffix}",
    )
    scaler = StandardScaler(
        inputCol=f"features_unscaled_{suffix}", outputCol=f"features_{suffix}", withStd=True, withMean=True
    )
    return indexers + encoders + [assembler, scaler]


def build_all_pipelines():
    """Construit les 5 pipelines ML."""
    # 1. Preprocessing Base (pour la procédure)
    prep_base = _build_preprocessing_stages(BASE_CATEGORICAL_COLS, BASE_NUMERIC_COLS, "base")
    
    # 2. Preprocessing Full (pour le reste)
    prep_full = _build_preprocessing_stages(FULL_CATEGORICAL_COLS, FULL_NUMERIC_COLS, "full")

    # Label indexers
    label_proc = StringIndexer(inputCol="type_procedure", outputCol="label_proc", handleInvalid="keep")
    label_statut = StringIndexer(inputCol="statut_final", outputCol="label_statut", handleInvalid="keep")
    label_action = StringIndexer(inputCol="next_best_action", outputCol="label_action", handleInvalid="keep")

    # Modèle 1 : Classification — Type Procédure (Automatisation totale)
    clf_proc = RandomForestClassifier(
        labelCol="label_proc", featuresCol="features_base", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_proc = Pipeline(stages=prep_base + [label_proc, clf_proc])

    # Modèle 2 : Classification — Probabilité de recouvrement
    clf_statut = RandomForestClassifier(
        labelCol="label_statut", featuresCol="features_full", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_statut = Pipeline(stages=prep_full + [label_statut, clf_statut])

    # Modèle 3 : Régression — Prédiction durée procédure
    reg_delai = RandomForestRegressor(
        labelCol="delai_final_jours", featuresCol="features_full", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_delai = Pipeline(stages=prep_full + [label_statut, reg_delai])

    # Modèle 4 : Classification — Next Best Action
    clf_action = RandomForestClassifier(
        labelCol="label_action", featuresCol="features_full", numTrees=50, maxDepth=5, seed=42
    )
    pipeline_action = Pipeline(stages=prep_full + [label_statut, label_action, clf_action])

    # Modèle 5 : Clustering — Segmentation
    kmeans = KMeans().setK(4).setSeed(42).setFeaturesCol("features_full").setPredictionCol("cluster_id")
    pipeline_cluster = Pipeline(stages=prep_full + [label_statut, kmeans])

    return pipeline_proc, pipeline_statut, pipeline_delai, pipeline_action, pipeline_cluster


def train_and_evaluate(df, p_proc, p_statut, p_delai, p_action, p_cluster):
    """Entraîne et évalue les 5 modèles."""
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # 1. Classification — Type Procédure
    logger.info("══ Modèle 1/5 : Classification (Type Procédure) ══")
    m_proc = p_proc.fit(train_data)
    acc_proc = MulticlassClassificationEvaluator(
        labelCol="label_proc", predictionCol="prediction", metricName="accuracy"
    ).evaluate(m_proc.transform(test_data))
    logger.info(f"   → Accuracy : {acc_proc:.4f}")

    # 2. Classification — Statut Final
    logger.info("══ Modèle 2/5 : Classification (Probabilité de recouvrement) ══")
    m_statut = p_statut.fit(train_data)
    acc = MulticlassClassificationEvaluator(
        labelCol="label_statut", predictionCol="prediction", metricName="accuracy"
    ).evaluate(m_statut.transform(test_data))
    logger.info(f"   → Accuracy : {acc:.4f}")

    # 3. Régression — Délai
    logger.info("══ Modèle 3/5 : Régression (Prédiction durée procédure) ══")
    m_delai = p_delai.fit(train_data)
    rmse = RegressionEvaluator(
        labelCol="delai_final_jours", predictionCol="prediction", metricName="rmse"
    ).evaluate(m_delai.transform(test_data))
    logger.info(f"   → RMSE : {rmse:.4f} jours")

    # 4. Classification — Next Best Action
    logger.info("══ Modèle 4/5 : Classification (Next Best Action) ══")
    m_action = p_action.fit(train_data)
    acc_action = MulticlassClassificationEvaluator(
        labelCol="label_action", predictionCol="prediction", metricName="accuracy"
    ).evaluate(m_action.transform(test_data))
    logger.info(f"   → Accuracy : {acc_action:.4f}")

    # 5. Clustering — KMeans
    logger.info("══ Modèle 5/5 : Clustering (Segmentation KMeans) ══")
    m_cluster = p_cluster.fit(df)
    silhouette = ClusteringEvaluator(
        featuresCol="features_full", predictionCol="cluster_id"
    ).evaluate(m_cluster.transform(df))
    logger.info(f"   → Silhouette : {silhouette:.4f}")

    return m_proc, m_statut, m_delai, m_action, m_cluster


def save_models(m_proc, m_statut, m_delai, m_action, m_cluster):
    """Sauvegarde les 5 modèles avec horodatage."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = BASE_DIR / "models"

    paths = {
        "rf_procedure": m_proc,
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
        p_proc, p_statut, p_delai, p_action, p_cluster = build_all_pipelines()
        m_proc, m_statut, m_delai, m_action, m_cluster = train_and_evaluate(
            df, p_proc, p_statut, p_delai, p_action, p_cluster
        )
        save_models(m_proc, m_statut, m_delai, m_action, m_cluster)
        logger.info("════ Entraînement terminé. 5 modèles sauvegardés avec succès. ════")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
