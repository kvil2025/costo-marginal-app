"""
Base Predictor - Clase abstracta para todos los modelos de predicción
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any

class BasePredictor(ABC):
    """Clase base abstracta para predictores de precios."""
    
    def __init__(self):
        self.is_trained = False
        self.training_data = None
        self.model_name = "Base"
    
    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena el modelo con datos históricos."""
        pass
    
    @abstractmethod
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera predicción para los próximos años."""
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara los datos para el modelo."""
        prophet_df = df[['timestamp', 'costo_marginal']].copy()
        prophet_df = prophet_df.rename(columns={
            'timestamp': 'ds',
            'costo_marginal': 'y'
        })
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.groupby(prophet_df['ds'].dt.date)['y'].mean().reset_index()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df[prophet_df['y'] > 0].dropna()
        return prophet_df
