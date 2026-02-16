import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import os
import sys
from datetime import datetime
from utils import get_db_engine

"""Pipeline de scroring para el proyecto."""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_scoring_dataset(engine, n_samples=10):
    """Crea dataset de scoring para muestras aleatorias del dataset"""
    
    logger.info(f"Creando dataset de scoring con {n_samples} muestras aleatorias.")
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scoring_dataset"))
        conn.commit()
        
        # Creo dataset de scoring con muestras aleatorias
        query = text(f"""
                     CREATE TABLE scoring_dataset AS
                     SELECT * FROM training_dataset
                     ORDER BY RANDOM()
                     LIMIT :n_samples
                     """)
        conn.execute(query, {"n_samples": n_samples})
        conn.commit()
        logger.info("Dataset de scoring creado exitosamente.")
        
        # Verifico que se hayan insertado las muestras
        result = conn.execute(text("SELECT COUNT(*) FROM scoring_dataset"))
        count = result.scalar()
        logger.info(f"Cantidad de muestras en scoring_dataset: {count}")
        return True

def load_scoring_data(engine):
    """Cargo el dataset de scoring desde la base de datos."""
    
    logger.info("Cargando dataset de scoring desde la base de datos.")
    
    query = text("SELECT * FROM scoring_dataset")
    df = pd.read_sql_query(query, engine)
    
    logger.info(f"Dataset de scoring cargado con {df.shape[0]} muestras.")
    
    return df

def load_training_model(model_path='models/best_model.pkl'):
    """Cargo el modelo previamente entrenado"""
    
    logger.info(f"Cargando modelo entrenado desde {model_path}.")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model=joblib.load(model_path)
    logger.info("Modelo cargado exitosamente.")
    logger.info(f"Tipo de modelo cargado: {type(model)}")
    
    return model


def main():
    """Pipeline principal de scoring."""
    
    try:
        logger.info("Iniciando pipeline de scoring.")
        
        #config
        n_samples= int(os.getenv('SCORING_SAMPLE_SIZE', '10'))
        
        # Paso 1: Conectando a la db
        logger.info("Conectando a la base de datos.")
        engine = get_db_engine()
        
        # Paso 2: Creando dataset de scoring
        create_scoring_dataset(engine, n_samples=n_samples)
        
        # Paso 3: Cargando dataset de scoring
        scoring_df = load_scoring_data(engine)
        
        # Paso 4: Cargando modelo entrenado
        model = load_training_model()
        
        
        return True
    
    except Exception as e:
        logger.error(f"Error en pipeline de scoring: str({e})", exc_info=True)
        return False
    
if __name__ == "__main__":
    succcess = main()
    sys.exit(0 if succcess else 1)