"""
ARIMA Predictor - Modelo ARIMA/SARIMA para predicción de precios
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.models.base_predictor import BasePredictor
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor(BasePredictor):
    """Predictor de precios usando ARIMA con auto-selección de parámetros."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = "ARIMA"
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena modelo ARIMA con auto-selección de parámetros."""
        try:
            from pmdarima import auto_arima
        except (ImportError, ValueError) as e:
            return {
                'success': False,
                'message': f'ARIMA no disponible: {str(e)[:100]}. Usa Prophet o XGBoost.'
            }
        
        prophet_df = self.prepare_data(df)
        
        if len(prophet_df) < 365:
            return {
                'success': False,
                'message': f'Datos insuficientes: {len(prophet_df)} días. Se necesitan al menos 365.'
            }
        
        self.training_data = prophet_df
        
        print(f"Entrenando ARIMA con {len(prophet_df)} días de datos...")
        
        # Auto ARIMA para encontrar mejores parámetros
        self.model = auto_arima(
            prophet_df['y'].values,
            start_p=1, start_q=1,
            max_p=5, max_q=5, max_d=2,
            seasonal=True, m=7,  # Estacionalidad semanal
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            d=None,  # Auto-detectar
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_fits=20
        )
        
        self.is_trained = True
        order = self.model.order
        seasonal = self.model.seasonal_order
        
        return {
            'success': True,
            'message': f'ARIMA{order}x{seasonal} entrenado con {len(prophet_df)} días',
            'date_range': f"{prophet_df['ds'].min().strftime('%Y-%m-%d')} a {prophet_df['ds'].max().strftime('%Y-%m-%d')}"
        }
    
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera predicción ARIMA."""
        if not self.is_trained:
            return pd.DataFrame(), {'success': False, 'message': 'Modelo no entrenado'}
        
        days_ahead = years_ahead * 365
        
        print(f"Generando predicción ARIMA para {years_ahead} años...")
        
        # Generar predicción con intervalos de confianza
        forecast, conf_int = self.model.predict(n_periods=days_ahead, return_conf_int=True)
        
        # Crear fechas futuras
        last_date = self.training_data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        # Combinar histórico + predicción
        hist_df = pd.DataFrame({
            'ds': self.training_data['ds'],
            'yhat': self.training_data['y'],
            'yhat_lower': self.training_data['y'],
            'yhat_upper': self.training_data['y'],
            'tipo': 'Histórico'
        })
        
        pred_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': np.clip(forecast, 0, None),
            'yhat_lower': np.clip(conf_int[:, 0], 0, None),
            'yhat_upper': np.clip(conf_int[:, 1], 0, None),
            'tipo': 'Predicción'
        })
        
        result = pd.concat([hist_df, pred_df], ignore_index=True)
        
        metrics = {
            'success': True,
            'prediction_start': str(last_date.date()),
            'prediction_end': str(future_dates[-1].date()),
            'avg_predicted_value': float(pred_df['yhat'].mean()),
            'min_predicted': float(pred_df['yhat'].min()),
            'max_predicted': float(pred_df['yhat'].max())
        }
        
        return result, metrics
    
    def get_seasonality_components(self):
        """ARIMA no tiene componentes de estacionalidad separados."""
        return None
