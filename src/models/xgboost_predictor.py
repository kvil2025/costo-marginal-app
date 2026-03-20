"""
XGBoost Predictor - Modelo de predicción de CMg basado en Gradient Boosting.
Reemplaza LSTM con un modelo más liviano, rápido y preciso para datos tabulares.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.models.base_predictor import BasePredictor
from src.data.feature_engineer import FeatureEngineer
import warnings
warnings.filterwarnings('ignore')


class XGBoostPredictor(BasePredictor):
    """Predictor de CMg usando XGBoost con feature engineering avanzado."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_upper = None  # Para intervalo superior
        self.model_lower = None  # Para intervalo inferior
        self.feature_engineer = FeatureEngineer()
        self.model_name = "XGBoost"
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.metrics_: Optional[Dict] = None
        self.engineered_data: Optional[pd.DataFrame] = None
    
    def _create_models(self):
        """Intenta crear modelos XGBoost, con fallback a sklearn."""
        try:
            import xgboost as xgb
            # Test that xgboost actually works (libomp check)
            test = xgb.XGBRegressor(n_estimators=1)
            test.fit(np.array([[1,2],[3,4]]), np.array([1,2]))
            self.use_xgb = True
            self.xgb_module = xgb
            print("  ✅ Usando XGBoost nativo")
        except (ImportError, Exception) as e:
            print(f"  ⚠️ XGBoost no disponible ({e}). Usando sklearn GradientBoosting como fallback.")
            self.use_xgb = False
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena modelo Gradient Boosting con feature engineering."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Detectar si XGBoost nativo está disponible
        self._create_models()
        
        # Preparar datos base
        prophet_df = self.prepare_data(df)
        
        if len(prophet_df) < 365:
            return {
                'success': False,
                'message': f'Datos insuficientes: {len(prophet_df)} días. Se necesitan al menos 365.'
            }
        
        self.training_data = prophet_df
        
        # Reconstruir a formato horario para feature engineering
        hourly_df = df[['timestamp', 'costo_marginal']].copy()
        hourly_df.columns = ['timestamp', 'costo_usd']
        hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
        hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)
        
        # Si hay duplicados por timestamp, promediar
        hourly_df = hourly_df.groupby('timestamp')['costo_usd'].mean().reset_index()
        
        # Agregar por día para el modelo (consistente con otros predictores)
        daily_df = hourly_df.set_index('timestamp').resample('D')['costo_usd'].mean().reset_index()
        daily_df = daily_df[daily_df['costo_usd'] > 0].dropna().reset_index(drop=True)
        
        engine_name = "XGBoost" if self.use_xgb else "GradientBoosting"
        print(f"🧠 Entrenando {engine_name} con {len(daily_df)} días de datos...")
        
        # Feature engineering
        self.engineered_data = self.feature_engineer.transform(daily_df, 'costo_usd')
        
        if len(self.engineered_data) < 200:
            return {
                'success': False,
                'message': f'Datos insuficientes después de feature engineering: {len(self.engineered_data)} días.'
            }
        
        # Split temporal
        splits = self.feature_engineer.prepare_train_test(
            self.engineered_data, 'costo_usd', test_ratio=0.15, val_ratio=0.1
        )
        
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_val = splits['X_val']
        y_val = splits['y_val']
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        if self.use_xgb:
            xgb = self.xgb_module
            params = {
                'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
                'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
            }
            self.model = xgb.XGBRegressor(objective='reg:squarederror', early_stopping_rounds=30, **params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.model_upper = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=0.975,
                early_stopping_rounds=30,
                **{k: v for k, v in params.items() if k != 'n_estimators'}, n_estimators=300)
            self.model_upper.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.model_lower = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=0.025,
                early_stopping_rounds=30,
                **{k: v for k, v in params.items() if k != 'n_estimators'}, n_estimators=300)
            self.model_lower.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # Fallback: sklearn GradientBoosting (no requiere libomp)
            self.model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
                loss='squared_error', validation_fraction=0.1,
                n_iter_no_change=20, tol=0.01
            )
            self.model.fit(X_train, y_train)
            
            # Quantile models for confidence intervals
            self.model_upper = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
                loss='quantile', alpha=0.975
            )
            self.model_upper.fit(X_train, y_train)
            
            self.model_lower = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
                loss='quantile', alpha=0.025
            )
            self.model_lower.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluar en test set
        y_pred = self.model.predict(X_test)
        mae = np.mean(np.abs(y_test.values - y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / np.maximum(y_test.values, 1))) * 100
        rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
        
        self.metrics_ = {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'rmse': round(rmse, 2),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
        }
        
        # Feature importance
        importance = self.model.feature_importances_
        features = splits['features']
        self.feature_importance_ = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"  ✅ MAE: ${mae:.2f}/MWh | MAPE: {mape:.1f}% | RMSE: ${rmse:.2f}/MWh")
        print(f"  📊 Top features: {', '.join(self.feature_importance_.head(5)['feature'].tolist())}")
        
        return {
            'success': True,
            'message': f'XGBoost entrenado con {len(daily_df)} días | MAE: ${mae:.2f} | MAPE: {mape:.1f}%',
            'date_range': f"{daily_df['timestamp'].min().strftime('%Y-%m-%d')} a {daily_df['timestamp'].max().strftime('%Y-%m-%d')}",
            'metrics': self.metrics_
        }
    
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera predicción con intervalos de confianza - optimizada por bloques."""
        if not self.is_trained or self.engineered_data is None:
            return pd.DataFrame(), {'success': False, 'message': 'Modelo no entrenado'}
        
        days_ahead = years_ahead * 365
        
        print(f"🔮 Generando predicción para {years_ahead} años ({days_ahead} días)...")
        
        last_data = self.engineered_data.copy()
        last_date = last_data['timestamp'].max()
        features = self.feature_engineer.get_feature_names()
        
        # Predicción por bloques de 30 días (mucho más rápido que día a día)
        BLOCK_SIZE = 30
        combined = last_data[['timestamp', 'costo_usd']].copy()
        days_predicted = 0
        
        while days_predicted < days_ahead:
            block = min(BLOCK_SIZE, days_ahead - days_predicted)
            next_dates = pd.date_range(
                start=combined['timestamp'].max() + pd.Timedelta(days=1),
                periods=block, freq='D'
            )
            
            # Usar último valor conocido como semilla
            last_val = combined['costo_usd'].iloc[-1]
            future_block = pd.DataFrame({
                'timestamp': next_dates,
                'costo_usd': last_val  # Initial seed
            })
            
            # Combinar con histórico reciente (500 días suficiente para features)
            tail_size = min(500, len(combined))
            window = pd.concat([combined.tail(tail_size), future_block], ignore_index=True)
            
            # Generar features una vez para todo el bloque
            window = self.feature_engineer.add_temporal_features(window)
            window = self.feature_engineer.add_lag_features(window, 'costo_usd')
            window = self.feature_engineer.add_rolling_features(window, 'costo_usd')
            window = window.replace([np.inf, -np.inf], np.nan)
            
            # Predecir todo el bloque de una vez
            future_rows = window.tail(block)
            X_future = future_rows[features].fillna(0)
            
            preds = np.clip(self.model.predict(X_future), 0, None)
            
            pred_block = pd.DataFrame({
                'timestamp': next_dates,
                'costo_usd': preds
            })
            combined = pd.concat([combined, pred_block], ignore_index=True)
            days_predicted += block
            
            if days_predicted % 365 == 0:
                print(f"  📅 {days_predicted // 365}/{years_ahead} años predichos...")
        
        print(f"  ✅ Predicción completada")
        
        # Intervalos de confianza
        combined_engineered = self.feature_engineer.transform(combined, 'costo_usd', drop_na=False)
        future_mask = combined_engineered['timestamp'] > last_date
        future_features = combined_engineered.loc[future_mask, features].fillna(0)
        
        if len(future_features) > 0:
            preds_upper = np.clip(self.model_upper.predict(future_features), 0, None)
            preds_lower = np.clip(self.model_lower.predict(future_features), 0, None)
        else:
            preds_upper = np.array([])
            preds_lower = np.array([])
        
        # Construir resultado
        hist_mask = combined_engineered['timestamp'] <= last_date
        hist_result = combined_engineered.loc[hist_mask]
        
        hist_df = pd.DataFrame({
            'ds': hist_result['timestamp'],
            'yhat': hist_result['costo_usd'],
            'yhat_lower': hist_result['costo_usd'],
            'yhat_upper': hist_result['costo_usd'],
            'tipo': 'Histórico'
        })
        
        future_result = combined_engineered.loc[future_mask]
        min_len = min(len(future_result), len(preds_upper), len(preds_lower))
        
        pred_df = pd.DataFrame({
            'ds': future_result['timestamp'].values[:min_len],
            'yhat': future_result['costo_usd'].values[:min_len],
            'yhat_lower': preds_lower[:min_len],
            'yhat_upper': preds_upper[:min_len],
            'tipo': 'Predicción'
        })
        
        result = pd.concat([hist_df, pred_df], ignore_index=True)
        
        # Métricas
        pred_values = pred_df['yhat']
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        metrics = {
            'success': True,
            'prediction_start': str(last_date.date()),
            'prediction_end': str(future_dates[-1].date()),
            'avg_predicted_value': float(pred_values.mean()) if len(pred_values) > 0 else 0,
            'min_predicted': float(pred_values.min()) if len(pred_values) > 0 else 0,
            'max_predicted': float(pred_values.max()) if len(pred_values) > 0 else 0,
            'model_metrics': self.metrics_
        }
        
        return result, metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Retorna importancia de features."""
        return self.feature_importance_
    
    def get_model_metrics(self) -> Optional[Dict]:
        """Retorna métricas de evaluación del modelo."""
        return self.metrics_
    
    def get_seasonality_components(self):
        """XGBoost no tiene componentes de estacionalidad separados."""
        return None
