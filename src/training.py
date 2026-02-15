import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import logging
import os
import sys
import json
from datetime import datetime
from utils import get_db_engine

# Setup de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def load_training_data(engine):
    """Cargo los datos de entrenamiento desde la db"""
    
    logger.info("Cargando datos de entrenamiento desde la base de datos...")
    
    query = "SELECT * FROM training_dataset";
    df = pd.read_sql(query, engine)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    logger.info(f"Columnas: {df.columns.tolist()}")
    
    return df

def prepare_features_target(df):
    """Separar features y target"""

    # Eliminar columnas no necesarias
    columns_to_drop = ['id', 'created_at']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

    # Separar X e y
    X = df.drop('charges', axis=1)
    y = df['charges']

    logger.info(f"Features (X): {X.columns.tolist()}")
    logger.info(f"Target (y): charges")
    logger.info(f"  - Min: ${y.min():,.2f}")
    logger.info(f"  - Max: ${y.max():,.2f}")
    logger.info(f"  - Mean: ${y.mean():,.2f}")
    logger.info(f"  - Median: ${y.median():,.2f}")

    return X, y

def split_data(X, y, test_size=0.2, random_state=43):
    """Split train/validation"""

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    logger.info(f"Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    logger.info(f"Validation set: {X_val.shape[0]} samples ({test_size*100:.0f}%)")
    logger.info(f"Train target - mean: ${y_train.mean():,.2f}, std: ${y_train.std():,.2f}")
    logger.info(f"Val target - mean: ${y_val.mean():,.2f}, std: ${y_val.std():,.2f}")

    return X_train, X_val, y_train, y_val

def main():
    """Función principal para ejecutar el proceso de entrenamiento."""
    
    try:
        logger.info("Iniciando proceso de entrenamiento...")

        # Paso 1: Conectar a la base de datos
        engine = get_db_engine()

        # Paso 2: Cargar datos de entrenamiento
        df = load_training_data(engine)
        logger.info("Proceso de entrenamiento completado.")
        
        # Paso 3: Preparar features y target
        logger.info("Preparando features y target...")
        X, y = prepare_features_target(df)
        
        # Paso 4: Split train/validation
        logger.info("Dividiendo datos en train y validation...")
        X_train, X_val, y_train, y_val = split_data(X, y)
        
        
        return True
    
    
    except Exception as e:
        logger.error(f"\nError en el proceso de entrenamiento: {str(e)}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)