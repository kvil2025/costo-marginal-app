"""
LSTM Predictor - Red neuronal LSTM para predicción de precios
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.models.base_predictor import BasePredictor
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor(BasePredictor):
    """Predictor de precios usando LSTM (Long Short-Term Memory)."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.sequence_length = 60  # 60 días de contexto
        self.model_name = "LSTM"
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena red LSTM."""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            
            # Suprimir logs de TensorFlow
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            
        except ImportError as e:
            return {
                'success': False,
                'message': f'TensorFlow no disponible: {str(e)}'
            }
        
        try:
            prophet_df = self.prepare_data(df)
            
            if len(prophet_df) < 365:
                return {
                    'success': False,
                    'message': f'Datos insuficientes: {len(prophet_df)} días.'
                }
            
            self.training_data = prophet_df
            
            print(f"Entrenando LSTM con {len(prophet_df)} días de datos...")
            
            # Normalizar datos
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(prophet_df['y'].values.reshape(-1, 1))
            
            # Crear secuencias para LSTM
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Construir modelo LSTM
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25),
                tf.keras.layers.Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Entrenar (pocas épocas para rapidez)
            self.model.fit(X, y, batch_size=32, epochs=10, verbose=0)
            
            self.is_trained = True
            
            return {
                'success': True,
                'message': f'LSTM entrenado con {len(prophet_df)} días',
                'date_range': f"{prophet_df['ds'].min().strftime('%Y-%m-%d')} a {prophet_df['ds'].max().strftime('%Y-%m-%d')}"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error entrenando LSTM: {str(e)[:100]}'
            }
    
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera predicción LSTM."""
        if not self.is_trained:
            return pd.DataFrame(), {'success': False, 'message': 'Modelo no entrenado'}
        
        days_ahead = years_ahead * 365
        
        print(f"Generando predicción LSTM para {years_ahead} años...")
        
        # Obtener últimos datos para iniciar predicción
        scaled_data = self.scaler.transform(self.training_data['y'].values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Predicción iterativa
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(pred[0, 0])
            # Actualizar secuencia
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]
        
        # Desnormalizar
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        predictions = np.clip(predictions, 0, None)
        
        # Crear fechas futuras
        last_date = self.training_data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        # Calcular intervalo de confianza (simple: ±15%)
        uncertainty = predictions * 0.15
        
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
            'yhat': predictions,
            'yhat_lower': predictions - uncertainty,
            'yhat_upper': predictions + uncertainty,
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
        """LSTM no tiene componentes de estacionalidad separados."""
        return None
