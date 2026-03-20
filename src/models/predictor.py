"""
Price Predictor - Modelo de predicción de precios usando Prophet
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class PricePredictor:
    """Predictor de precios de costo marginal usando Prophet."""
    
    def __init__(self):
        self.model: Optional[Prophet] = None
        self.is_trained = False
        self.training_data: Optional[pd.DataFrame] = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara los datos para Prophet (requiere columnas 'ds' y 'y')."""
        prophet_df = df[['timestamp', 'costo_marginal']].copy()
        prophet_df = prophet_df.rename(columns={
            'timestamp': 'ds',
            'costo_marginal': 'y'
        })
        
        # Agregar por día para reducir ruido
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.groupby(prophet_df['ds'].dt.date)['y'].mean().reset_index()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Eliminar valores negativos o nulos
        prophet_df = prophet_df[prophet_df['y'] > 0].dropna()
        
        return prophet_df
    
    def train(self, df: pd.DataFrame) -> dict:
        """Entrena el modelo con datos históricos."""
        prophet_df = self.prepare_data(df)
        
        if len(prophet_df) < 365:
            return {
                'success': False,
                'message': f'Datos insuficientes: {len(prophet_df)} días. Se necesitan al menos 365 días.'
            }
        
        self.training_data = prophet_df
        
        # Configurar Prophet con estacionalidades para datos eléctricos
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # Agregamos por día, no necesitamos diaria
            changepoint_prior_scale=0.05,  # Flexibilidad moderada
            seasonality_prior_scale=10.0,
            interval_width=0.95  # Intervalo de confianza del 95%
        )
        
        # Agregar estacionalidad mensual personalizada
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        print(f"Entrenando modelo con {len(prophet_df)} días de datos...")
        self.model.fit(prophet_df)
        self.is_trained = True
        
        return {
            'success': True,
            'message': f'Modelo entrenado con {len(prophet_df)} días de datos',
            'date_range': f"{prophet_df['ds'].min().strftime('%Y-%m-%d')} a {prophet_df['ds'].max().strftime('%Y-%m-%d')}"
        }
    
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, dict]:
        """Genera predicción para los próximos años."""
        if not self.is_trained:
            return pd.DataFrame(), {'success': False, 'message': 'Modelo no entrenado'}
        
        # Crear dataframe de fechas futuras
        days_ahead = years_ahead * 365
        future = self.model.make_future_dataframe(periods=days_ahead, freq='D')
        
        # Generar predicción
        print(f"Generando predicción para {years_ahead} años...")
        forecast = self.model.predict(future)
        
        # Separar datos históricos y predicción
        last_historical_date = self.training_data['ds'].max()
        forecast['tipo'] = forecast['ds'].apply(
            lambda x: 'Histórico' if x <= last_historical_date else 'Predicción'
        )
        
        # Asegurar que los valores sean positivos
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        metrics = {
            'success': True,
            'prediction_start': str(last_historical_date.date()),
            'prediction_end': str(forecast['ds'].max().date()),
            'avg_predicted_value': float(forecast[forecast['tipo'] == 'Predicción']['yhat'].mean()),
            'min_predicted': float(forecast[forecast['tipo'] == 'Predicción']['yhat'].min()),
            'max_predicted': float(forecast[forecast['tipo'] == 'Predicción']['yhat'].max())
        }
        
        return forecast, metrics
    
    def get_seasonality_components(self) -> Optional[pd.DataFrame]:
        """Obtiene los componentes de estacionalidad del modelo."""
        if not self.is_trained:
            return None
        
        # Crear fechas para un año completo
        future = self.model.make_future_dataframe(periods=365, freq='D')
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'trend', 'yearly', 'weekly', 'monthly']]
    
    def evaluate_model(self, test_days: int = 365) -> dict:
        """Evalúa el modelo usando los últimos días como conjunto de prueba."""
        if not self.is_trained or self.training_data is None:
            return {'success': False, 'message': 'Modelo no entrenado'}
        
        if len(self.training_data) < test_days * 2:
            return {'success': False, 'message': 'Datos insuficientes para evaluación'}
        
        # Dividir datos
        train_df = self.training_data.iloc[:-test_days]
        test_df = self.training_data.iloc[-test_days:]
        
        # Entrenar modelo temporal
        temp_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        temp_model.fit(train_df)
        
        # Predecir
        forecast = temp_model.predict(test_df[['ds']])
        
        # Calcular métricas
        actual = test_df['y'].values
        predicted = forecast['yhat'].values
        
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {
            'success': True,
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),
            'test_days': test_days
        }
