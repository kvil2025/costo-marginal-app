"""
Feature Engineering para predicción de Costo Marginal.
Genera features temporales, lags, rolling stats e inter-barras.
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Pipeline de feature engineering para series temporales de CMg."""
    
    FEATURE_GROUPS = {
        'temporal': [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
            'sin_dow', 'cos_dow'
        ],
        'lags': [
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h',
            'lag_24h', 'lag_48h', 'lag_168h', 'lag_720h'
        ],
        'rolling': [
            'rolling_mean_6h', 'rolling_std_6h',
            'rolling_mean_24h', 'rolling_std_24h',
            'rolling_min_24h', 'rolling_max_24h',
            'rolling_mean_7d', 'rolling_std_7d',
            'rolling_mean_30d',
            'price_change_1h', 'price_change_24h', 'price_change_7d'
        ],
    }
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega features basadas en el timestamp."""
        df = df.copy()
        ts = df['timestamp']
        
        # Features directas
        df['hour'] = ts.dt.hour
        df['day_of_week'] = ts.dt.dayofweek
        df['day_of_month'] = ts.dt.day
        df['month'] = ts.dt.month
        df['quarter'] = ts.dt.quarter
        df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
        
        # Codificación cíclica (mejor que one-hot para datos circulares)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'costo_usd') -> pd.DataFrame:
        """Agrega valores pasados del target como features."""
        df = df.copy()
        
        lag_periods = {
            'lag_1h': 1,
            'lag_2h': 2,
            'lag_3h': 3,
            'lag_6h': 6,
            'lag_12h': 12,
            'lag_24h': 24,       # Ayer misma hora
            'lag_48h': 48,       # 2 días atrás
            'lag_168h': 168,     # Semana pasada misma hora
            'lag_720h': 720,     # ~1 mes atrás
        }
        
        for name, periods in lag_periods.items():
            df[name] = df[target_col].shift(periods)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str = 'costo_usd') -> pd.DataFrame:
        """Agrega estadísticas de ventana móvil."""
        df = df.copy()
        cost = df[target_col]
        
        # Rolling 6 horas
        df['rolling_mean_6h'] = cost.rolling(6, min_periods=1).mean()
        df['rolling_std_6h'] = cost.rolling(6, min_periods=1).std()
        
        # Rolling 24 horas
        df['rolling_mean_24h'] = cost.rolling(24, min_periods=1).mean()
        df['rolling_std_24h'] = cost.rolling(24, min_periods=1).std()
        df['rolling_min_24h'] = cost.rolling(24, min_periods=1).min()
        df['rolling_max_24h'] = cost.rolling(24, min_periods=1).max()
        
        # Rolling 7 días
        df['rolling_mean_7d'] = cost.rolling(168, min_periods=1).mean()
        df['rolling_std_7d'] = cost.rolling(168, min_periods=1).std()
        
        # Rolling 30 días
        df['rolling_mean_30d'] = cost.rolling(720, min_periods=1).mean()
        
        # Cambios porcentuales
        df['price_change_1h'] = cost.pct_change(1)
        df['price_change_24h'] = cost.pct_change(24)
        df['price_change_7d'] = cost.pct_change(168)
        
        return df
    
    def transform(self, df: pd.DataFrame, target_col: str = 'costo_usd',
                  drop_na: bool = True) -> pd.DataFrame:
        """Pipeline completo de feature engineering."""
        # Asegurar que está ordenado por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Agregar todas las features
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df, target_col)
        df = self.add_rolling_features(df, target_col)
        
        # Reemplazar infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Guardar nombres de features
        self.feature_names = (
            self.FEATURE_GROUPS['temporal'] +
            self.FEATURE_GROUPS['lags'] +
            self.FEATURE_GROUPS['rolling']
        )
        
        if drop_na:
            # Eliminar filas con NaN en features (primeras ~720 horas por los lags)
            df = df.dropna(subset=self.feature_names).reset_index(drop=True)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nombres de features generadas."""
        return self.feature_names
    
    def prepare_train_test(self, df: pd.DataFrame, target_col: str = 'costo_usd',
                           test_ratio: float = 0.2, val_ratio: float = 0.1):
        """Divide datos en train/val/test cronológicamente."""
        n = len(df)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        features = self.get_feature_names()
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_val = val_df[features]
        y_val = val_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_dates': train_df['timestamp'],
            'val_dates': val_df['timestamp'],
            'test_dates': test_df['timestamp'],
            'features': features
        }
