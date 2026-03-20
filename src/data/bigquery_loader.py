"""
BigQuery Data Loader para CMARG Pro
Maneja todas las consultas a BigQuery para el dashboard.
Con fallback automático a CSV local para desarrollo.
"""
import os
import pandas as pd
from pathlib import Path
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Intentar importar BigQuery
try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except ImportError:
    BQ_AVAILABLE = False


class BigQueryLoader:
    """Loader de datos desde BigQuery con fallback a CSV local."""
    
    # Proyecto y dataset de BigQuery
    PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'geologgia-map')
    DATASET = os.environ.get('BQ_DATASET', 'cmarg')
    
    # Tablas
    TABLE_COSTOS = f'{PROJECT_ID}.{DATASET}.costos_marginales'
    TABLE_EXOGENOS = f'{PROJECT_ID}.{DATASET}.datos_exogenos'
    TABLE_BARRAS = f'{PROJECT_ID}.{DATASET}.barras_metadata'
    
    def __init__(self):
        self.use_bigquery = os.environ.get('USE_BIGQUERY', 'false').lower() == 'true'
        self.client = None
        self._cache = {}
        
        if self.use_bigquery and BQ_AVAILABLE:
            try:
                self.client = bigquery.Client(project=self.PROJECT_ID)
                print("✅ Conectado a BigQuery")
            except Exception as e:
                print(f"⚠️ Error conectando a BigQuery: {e}")
                print("📁 Usando CSV local como fallback")
                self.use_bigquery = False
        else:
            if not BQ_AVAILABLE:
                print("📁 google-cloud-bigquery no instalado — usando CSV local")
            else:
                print("📁 USE_BIGQUERY=false — usando CSV local")
    
    def get_barras_disponibles(self):
        """Retorna lista de barras disponibles con metadata."""
        if self.use_bigquery and self.client:
            return self._get_barras_bq()
        return self._get_barras_csv()
    
    def get_barra_data(self, barra_key: str) -> pd.DataFrame:
        """Obtiene datos históricos de una barra específica."""
        if self.use_bigquery and self.client:
            return self._get_barra_data_bq(barra_key)
        return self._get_barra_data_csv(barra_key)
    
    def get_exogenous_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Obtiene datos exógenos (económicos/clima)."""
        if self.use_bigquery and self.client:
            return self._get_exogenous_bq(start_date, end_date)
        return self._get_exogenous_csv()
    
    def get_stats(self, barra_key: str) -> dict:
        """Obtiene estadísticas rápidas de una barra."""
        if self.use_bigquery and self.client:
            return self._get_stats_bq(barra_key)
        # Para CSV, calculamos desde los datos cargados
        return None  # El app.py calcula las stats desde el DataFrame
    
    # =========================================================================
    # BigQuery Methods
    # =========================================================================
    
    def _get_barras_bq(self):
        """Lista barras desde BigQuery."""
        query = f"""
        SELECT DISTINCT 
            barra_mnemotecnico,
            barra_nombre,
            MIN(fecha) as fecha_inicio,
            MAX(fecha) as fecha_fin,
            COUNT(*) as registros
        FROM `{self.TABLE_COSTOS}`
        GROUP BY barra_mnemotecnico, barra_nombre
        ORDER BY barra_nombre
        """
        try:
            df = self.client.query(query).to_dataframe()
            return df
        except Exception as e:
            print(f"⚠️ Error en BigQuery: {e}")
            return self._get_barras_csv()
    
    def _get_barra_data_bq(self, barra_key: str) -> pd.DataFrame:
        """Obtiene datos de una barra desde BigQuery."""
        # Cache key
        cache_key = f"barra_{barra_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        query = f"""
        SELECT 
            timestamp,
            costo_usd as costo_marginal,
            costo_clp,
            barra_mnemotecnico,
            barra_nombre
        FROM `{self.TABLE_COSTOS}`
        WHERE barra_mnemotecnico = @barra_key
        ORDER BY timestamp
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("barra_key", "STRING", barra_key)
            ]
        )
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self._cache[cache_key] = df
            return df
        except Exception as e:
            print(f"⚠️ Error en BigQuery: {e}")
            return self._get_barra_data_csv(barra_key)
    
    def _get_exogenous_bq(self, start_date=None, end_date=None) -> pd.DataFrame:
        """Obtiene datos exógenos desde BigQuery."""
        where_clause = ""
        params = []
        
        if start_date:
            where_clause += " WHERE fecha >= @start_date"
            params.append(bigquery.ScalarQueryParameter("start_date", "DATE", start_date))
        if end_date:
            if where_clause:
                where_clause += " AND fecha <= @end_date"
            else:
                where_clause += " WHERE fecha <= @end_date"
            params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date))
        
        query = f"""
        SELECT *
        FROM `{self.TABLE_EXOGENOS}`
        {where_clause}
        ORDER BY fecha
        """
        
        job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
        
        try:
            df = self.client.query(query, job_config=job_config).to_dataframe()
            return df
        except Exception as e:
            print(f"⚠️ Error en BigQuery: {e}")
            return self._get_exogenous_csv()
    
    def _get_stats_bq(self, barra_key: str) -> dict:
        """Obtiene estadísticas rápidas desde BigQuery."""
        query = f"""
        SELECT 
            COUNT(*) as registros,
            MIN(fecha) as fecha_inicio,
            MAX(fecha) as fecha_fin,
            AVG(costo_usd) as promedio_usd,
            MIN(costo_usd) as min_usd,
            MAX(costo_usd) as max_usd,
            STDDEV(costo_usd) / AVG(costo_usd) * 100 as volatilidad
        FROM `{self.TABLE_COSTOS}`
        WHERE barra_mnemotecnico = @barra_key
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("barra_key", "STRING", barra_key)
            ]
        )
        
        try:
            result = self.client.query(query, job_config=job_config).to_dataframe()
            if len(result) > 0:
                row = result.iloc[0]
                return {
                    'registros': int(row['registros']),
                    'fecha_inicio': row['fecha_inicio'],
                    'fecha_fin': row['fecha_fin'],
                    'promedio_usd': float(row['promedio_usd']),
                    'min_usd': float(row['min_usd']),
                    'max_usd': float(row['max_usd']),
                    'volatilidad': float(row['volatilidad'])
                }
        except Exception as e:
            print(f"⚠️ Error en BigQuery stats: {e}")
        return None
    
    # =========================================================================
    # CSV Fallback Methods (para desarrollo local)
    # =========================================================================
    
    def _get_barras_csv(self):
        """Lista barras disponibles desde archivos CSV locales."""
        barras_dir = Path(__file__).parent.parent.parent / 'data' / 'barras'
        barras = []
        
        for csv_file in sorted(barras_dir.glob('*.csv')):
            if csv_file.name in ['all_barras.csv', 'extraction_stats.csv']:
                continue
            name = csv_file.stem.replace('_', ' ').title()
            barras.append({
                'file': csv_file.name,
                'key': csv_file.stem,
                'nombre': name
            })
        return barras
    
    def _get_barra_data_csv(self, barra_key: str) -> pd.DataFrame:
        """Carga datos de una barra desde CSV local."""
        barras_dir = Path(__file__).parent.parent.parent / 'data' / 'barras'
        csv_path = barras_dir / f"{barra_key}.csv"
        
        if not csv_path.exists():
            # Try to find by matching name
            for f in barras_dir.glob('*.csv'):
                if barra_key in f.stem:
                    csv_path = f
                    break
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'fecha' in df.columns and 'hora' in df.columns:
                df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'], unit='h')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if 'costo_usd' in df.columns and 'costo_marginal' not in df.columns:
                df['costo_marginal'] = df['costo_usd']
            
            return df
        
        return pd.DataFrame()
    
    def _get_exogenous_csv(self) -> pd.DataFrame:
        """Carga datos exógenos desde CSV local."""
        exo_path = Path(__file__).parent.parent.parent / 'data' / 'exogenous' / 'exogenous_data.csv'
        if exo_path.exists():
            df = pd.read_csv(exo_path)
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
            return df
        return pd.DataFrame()
    
    def clear_cache(self):
        """Limpia la caché de consultas."""
        self._cache = {}
