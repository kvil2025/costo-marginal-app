"""
XGBoost Predictor V2 - Modelo REALMENTE predictivo.

Diferencias vs V1:
- Dominado por calendar features (no lags autorecursivos)
- Integra variables exógenas (dólar, cobre, El Niño/La Niña)
- Predicción corta (7 días) con alta confianza
- Predicción larga (3 meses) con confianza moderada
- NO repite patrones - captura relaciones causales
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from src.models.base_predictor import BasePredictor
import warnings
import os
warnings.filterwarnings('ignore')


class XGBoostPredictorV2(BasePredictor):
    """Predictor con variables exógenas y calendar features dominantes."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_upper = None
        self.model_lower = None
        self.model_name = "XGBoost V2"
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.metrics_: Optional[Dict] = None
        self.features_list: List[str] = []
        self.exogenous_data: Optional[pd.DataFrame] = None
        self.daily_data: Optional[pd.DataFrame] = None
        self.last_date = None
    
    def _load_exogenous(self):
        """Carga datos exógenos (dólar, cobre, ONI, etc.)"""
        exo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'exogenous', 'exogenous_data.csv'
        )
        if os.path.exists(exo_path):
            df = pd.read_csv(exo_path, parse_dates=['fecha'])
            print(f"  📊 Datos exógenos cargados: {len(df)} días, {df.shape[1]} variables")
            return df
        else:
            print(f"  ⚠️ No se encontró datos exógenos en {exo_path}")
            return None
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye features PREDICTIVAS (no autorecursivas)."""
        df = df.copy()
        ts = df['timestamp']
        
        # === CALENDAR FEATURES (siempre disponibles para futuro) ===
        df['hour'] = ts.dt.hour
        df['day_of_week'] = ts.dt.dayofweek
        df['day_of_month'] = ts.dt.day
        df['month'] = ts.dt.month
        df['quarter'] = ts.dt.quarter
        df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
        df['day_of_year'] = ts.dt.dayofyear
        df['week_of_year'] = ts.dt.isocalendar().week.astype(int)
        
        # Codificación cíclica (mejor para patrones del calendario)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Features especiales del mercado chileno
        df['is_peak_hour'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        df['is_solar_hour'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
        df['is_madrugada'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        
        # Interacción estación x hora (verano solar vs invierno)
        df['summer_solar'] = (df['is_solar_hour'] * 
                              ((df['month'] >= 10) | (df['month'] <= 3)).astype(int))
        df['winter_peak'] = (df['is_peak_hour'] * 
                             ((df['month'] >= 5) & (df['month'] <= 8)).astype(int))
        
        # Tendencia temporal (captura la penetración creciente de renovables)
        ref_date = pd.Timestamp('2019-01-01')
        df['trend_years'] = (ts - ref_date).dt.days / 365.25
        df['trend_years_sq'] = df['trend_years'] ** 2  # Efecto acelerado de renovables
        
        return df
    
    def _merge_exogenous(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge features exógenas con datos principales."""
        if self.exogenous_data is None:
            return df
        
        df = df.copy()
        df['fecha_merge'] = df['timestamp'].dt.date
        
        exo = self.exogenous_data.copy()
        exo['fecha_merge'] = exo['fecha'].dt.date
        
        # Seleccionar columnas a merge
        exo_cols = ['fecha_merge']
        available_cols = []
        for col in ['usd_clp', 'cobre_usd', 'oni_index', 'is_el_nino', 'is_la_nina',
                     'usd_clp_change_7d', 'usd_clp_change_30d', 'cobre_change_30d',
                     'oni_squared', 'eur_clp']:
            if col in exo.columns:
                exo_cols.append(col)
                available_cols.append(col)
        
        if available_cols:
            exo_subset = exo[exo_cols].drop_duplicates('fecha_merge')
            df = df.merge(exo_subset, on='fecha_merge', how='left')
            
            # Forward fill para datos faltantes
            for col in available_cols:
                df[col] = df[col].ffill().bfill()
            
            print(f"  ✅ Variables exógenas integradas: {available_cols}")
        
        df = df.drop('fecha_merge', axis=1)
        return df
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Entrena modelo con features predictivas + datos exógenos."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Cargar datos exógenos
        self.exogenous_data = self._load_exogenous()
        
        # Preparar datos base
        prophet_df = self.prepare_data(df)
        
        if len(prophet_df) < 365:
            return {
                'success': False,
                'message': f'Datos insuficientes: {len(prophet_df)} días.'
            }
        
        self.training_data = prophet_df
        
        # Trabajar con datos diarios
        hourly_df = df[['timestamp', 'costo_marginal']].copy()
        hourly_df.columns = ['timestamp', 'costo_usd']
        hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
        hourly_df = hourly_df.groupby('timestamp')['costo_usd'].mean().reset_index()
        
        daily_df = hourly_df.set_index('timestamp').resample('D')['costo_usd'].mean().reset_index()
        daily_df = daily_df[daily_df['costo_usd'] > 0].dropna().reset_index(drop=True)
        
        self.daily_data = daily_df.copy()
        self.last_date = daily_df['timestamp'].max()
        
        print(f"🧠 Entrenando XGBoost V2 con {len(daily_df)} días de datos...")
        print(f"   Rango: {daily_df['timestamp'].min().date()} → {daily_df['timestamp'].max().date()}")
        
        # === BUILD FEATURES ===
        featured = self._build_features(daily_df)
        featured = self._merge_exogenous(featured)
        
        # NO lag features - el modelo aprende patrones causales, no autorecursivos
        # Esto produce predicciones con variación estacional REAL, no flat lines
        
        # Drop NaN
        featured = featured.replace([np.inf, -np.inf], np.nan)
        featured = featured.dropna().reset_index(drop=True)
        
        if len(featured) < 200:
            return {'success': False, 'message': f'Datos insuficientes: {len(featured)} días.'}
        
        # Define features - SOLO calendar + exógenas (no lags)
        self.features_list = [c for c in featured.columns 
                              if c not in ['timestamp', 'costo_usd', 'fecha_merge']]
        
        print(f"  📋 {len(self.features_list)} features: {self.features_list[:10]}...")
        
        # Split temporal (85% train, 15% test)
        n = len(featured)
        train_size = int(n * 0.85)
        train_df = featured.iloc[:train_size]
        test_df = featured.iloc[train_size:]
        
        X_train = train_df[self.features_list]
        y_train = train_df['costo_usd']
        X_test = test_df[self.features_list]
        y_test = test_df['costo_usd']
        
        # Detectar XGBoost
        try:
            import xgboost as xgb
            test_model = xgb.XGBRegressor(n_estimators=1)
            test_model.fit(np.array([[1,2],[3,4]]), np.array([1,2]))
            use_xgb = True
            print("  ✅ Usando XGBoost nativo")
        except:
            use_xgb = False
            print("  ℹ️ Usando sklearn GradientBoosting")
        
        # Train models
        if use_xgb:
            params = {
                'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.03,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 10,
                'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': 42, 'n_jobs': -1,
            }
            # Validation split from train
            val_size = int(len(X_train) * 0.1)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            
            self.model = xgb.XGBRegressor(objective='reg:squarederror', 
                                           early_stopping_rounds=30, **params)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.model_upper = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=0.975,
                early_stopping_rounds=30,
                **{k: v for k, v in params.items() if k != 'n_estimators'}, n_estimators=400)
            self.model_upper.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            self.model_lower = xgb.XGBRegressor(
                objective='reg:quantileerror', quantile_alpha=0.025,
                early_stopping_rounds=30,
                **{k: v for k, v in params.items() if k != 'n_estimators'}, n_estimators=400)
            self.model_lower.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=15, random_state=42,
                validation_fraction=0.1, n_iter_no_change=30, tol=0.01
            )
            self.model.fit(X_train, y_train)
            
            self.model_upper = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=15, random_state=42,
                loss='quantile', alpha=0.975
            )
            self.model_upper.fit(X_train, y_train)
            
            self.model_lower = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                subsample=0.8, min_samples_leaf=15, random_state=42,
                loss='quantile', alpha=0.025
            )
            self.model_lower.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = np.mean(np.abs(y_test.values - y_pred))
        mape = np.mean(np.abs((y_test.values - y_pred) / np.maximum(y_test.values, 1))) * 100
        rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
        
        self.metrics_ = {
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'rmse': round(rmse, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.features_list),
            'has_exogenous': self.exogenous_data is not None,
        }
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance_ = pd.DataFrame({
            'feature': self.features_list,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"  ✅ MAE: ${mae:.2f}/MWh | MAPE: {mape:.1f}% | RMSE: ${rmse:.2f}/MWh")
        print(f"  📊 Top features:")
        for _, row in self.feature_importance_.head(8).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
        
        return {
            'success': True,
            'message': f'XGBoost V2 entrenado | {len(self.features_list)} features | MAE: ${mae:.2f} | MAPE: {mape:.1f}%',
            'date_range': f"{daily_df['timestamp'].min().strftime('%Y-%m-%d')} a {daily_df['timestamp'].max().strftime('%Y-%m-%d')}",
            'metrics': self.metrics_
        }
    
    def predict(self, years_ahead: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera predicción basada en calendar features + exógenas."""
        if not self.is_trained:
            return pd.DataFrame(), {'success': False, 'message': 'Modelo no entrenado'}
        
        # Calcular horizonte de predicción
        days_ahead = years_ahead * 365
        print(f"🔮 Generando predicción para {years_ahead} años ({days_ahead} días)...")
        if years_ahead > 3:
            print(f"   ℹ️ Horizonte largo: los intervalos de confianza se amplían con el tiempo")
        
        # Generar fechas futuras
        future_dates = pd.date_range(
            start=self.last_date + pd.Timedelta(days=1),
            periods=days_ahead, freq='D'
        )
        
        future_df = pd.DataFrame({'timestamp': future_dates})
        
        # Calendar features (siempre disponibles para el futuro)
        future_df = self._build_features(future_df)
        
        # Exogenous features para el futuro
        future_df = self._merge_exogenous(future_df)
        
        # Ensure all training features exist
        for feat in self.features_list:
            if feat not in future_df.columns:
                future_df[feat] = 0
        
        X_future = future_df[self.features_list].fillna(0)
        
        # Predict
        preds = np.clip(self.model.predict(X_future), 0, None)
        preds_upper = np.clip(self.model_upper.predict(X_future), 0, None)
        preds_lower = np.clip(self.model_lower.predict(X_future), 0, None)
        
        # Expandir intervalos con el tiempo (reconocer incertidumbre creciente)
        uncertainty_growth = np.linspace(1.0, 2.5, len(future_df))
        center = preds
        width_upper = (preds_upper - preds) * uncertainty_growth
        width_lower = (preds - preds_lower) * uncertainty_growth
        preds_upper = np.clip(center + width_upper, 0, None)
        preds_lower = np.clip(center - width_lower, 0, None)
        
        # Build result dataframe
        hist_df = pd.DataFrame({
            'ds': self.daily_data['timestamp'],
            'yhat': self.daily_data['costo_usd'],
            'yhat_lower': self.daily_data['costo_usd'],
            'yhat_upper': self.daily_data['costo_usd'],
            'tipo': 'Histórico'
        })
        
        pred_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': preds,
            'yhat_lower': preds_lower,
            'yhat_upper': preds_upper,
            'tipo': 'Predicción'
        })
        
        result = pd.concat([hist_df, pred_df], ignore_index=True)
        
        metrics = {
            'success': True,
            'prediction_start': str(self.last_date.date()),
            'prediction_end': str(future_dates[-1].date()),
            'prediction_days': days_ahead,
            'avg_predicted_value': float(preds.mean()),
            'min_predicted': float(preds.min()),
            'max_predicted': float(preds.max()),
            'model_metrics': self.metrics_,
            'has_exogenous': self.exogenous_data is not None,
        }
        
        print(f"  ✅ Predicción: promedio ${preds.mean():.2f}/MWh")
        print(f"     Rango: ${preds.min():.2f} - ${preds.max():.2f}/MWh")
        
        return result, metrics
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self.feature_importance_
    
    def get_model_metrics(self) -> Optional[Dict]:
        return self.metrics_
    
    def get_seasonality_components(self):
        return None
