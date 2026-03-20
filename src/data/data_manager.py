"""
Data Manager - Carga optimizada de datos de Costo Marginal
Indexa todas las barras disponibles y carga datos por demanda.
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
import pickle

class DataManager:
    """Gestor de datos optimizado para grandes volúmenes de archivos TSV."""
    
    COLUMNS_OLD = ['codigo_central', 'fecha', 'hora', 'costo_marginal', 'nombre_central']
    COLUMNS_NEW = ['barra_mnemotecnico', 'barra_referencia', 'fecha', 'hora', 
                   'costo_dolares', 'costo_pesos', 'nombre']
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.bar_index: Dict[str, List[Path]] = {}
        self.cache_file = self.base_dir / '.bar_index.pkl'
        
    def build_index(self, force_rebuild: bool = False) -> Dict[str, str]:
        """Construye un índice de todas las barras disponibles."""
        if not force_rebuild and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.bar_index = pickle.load(f)
            return {name: name for name in sorted(self.bar_index.keys())}
        
        print("Construyendo índice de barras (esto puede tomar unos minutos)...")
        all_files = list(self.base_dir.glob('**/*.tsv'))
        
        bar_names = set()
        file_mapping = {}
        
        for file in tqdm(all_files, desc="Indexando archivos"):
            try:
                # Leer solo la columna de nombres para indexar
                df = pd.read_csv(file, sep='\t', encoding='latin1', 
                                nrows=1000, usecols=[6] if 'DATA_' in str(file) else [6])
                col_name = df.columns[0]
                
                # Determinar si es formato antiguo o nuevo
                sample_df = pd.read_csv(file, sep='\t', encoding='latin1', 
                                       nrows=100, header=0)
                if 'nombre' in sample_df.columns:
                    names_col = 'nombre'
                elif 'nombre_central' in sample_df.columns:
                    names_col = 'nombre_central'
                else:
                    names_col = sample_df.columns[-1]
                
                unique_names = sample_df[names_col].unique()
                for name in unique_names:
                    if pd.notna(name):
                        bar_names.add(str(name))
                        if name not in file_mapping:
                            file_mapping[name] = []
                        file_mapping[name].append(file)
            except Exception as e:
                print(f"Error procesando {file}: {e}")
                continue
        
        self.bar_index = file_mapping
        
        # Guardar cache
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.bar_index, f)
        
        print(f"Índice construido: {len(bar_names)} barras únicas encontradas")
        return {name: name for name in sorted(bar_names)}
    
    def get_bar_options(self) -> List[Dict[str, str]]:
        """Retorna lista de opciones para dropdown."""
        if not self.bar_index:
            self.build_index()
        return [{'label': name, 'value': name} for name in sorted(self.bar_index.keys())]
    
    def load_bar_data(self, bar_name: str) -> pd.DataFrame:
        """Carga todos los datos históricos de una barra específica."""
        all_files = list(self.base_dir.glob('**/*.tsv'))
        
        dfs = []
        for file in tqdm(all_files, desc=f"Cargando datos de {bar_name[:30]}..."):
            try:
                # Leer archivo completo
                df = pd.read_csv(file, sep='\t', encoding='latin1', decimal=',')
                
                # Determinar columna de nombre
                if 'nombre' in df.columns:
                    name_col = 'nombre'
                    cost_col = 'costo_en_dolares'
                elif 'nombre_central' in df.columns:
                    name_col = 'nombre_central'
                    cost_col = 'costo_marginal' if 'costo_marginal' in df.columns else 'costo_en_dolares'
                else:
                    continue
                
                # Filtrar por barra
                df_filtered = df[df[name_col] == bar_name].copy()
                
                if df_filtered.empty:
                    continue
                
                # Normalizar columnas
                df_filtered = df_filtered.rename(columns={
                    name_col: 'nombre_central',
                    cost_col: 'costo_marginal'
                })
                
                # Crear timestamp
                if 'fecha' in df_filtered.columns and 'hora' in df_filtered.columns:
                    df_filtered['timestamp'] = pd.to_datetime(df_filtered['fecha']) + \
                                               pd.to_timedelta(df_filtered['hora'].astype(int) - 1, unit='h')
                
                dfs.append(df_filtered[['timestamp', 'costo_marginal', 'nombre_central']])
                
            except Exception as e:
                continue
        
        if not dfs:
            return pd.DataFrame(columns=['timestamp', 'costo_marginal', 'nombre_central'])
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values('timestamp').reset_index(drop=True)
        result = result.drop_duplicates(subset=['timestamp'])
        
        # Convertir costo_marginal a numérico
        result['costo_marginal'] = pd.to_numeric(result['costo_marginal'], errors='coerce')
        
        return result
    
    def get_quick_bar_list(self) -> List[str]:
        """Obtiene una lista rápida de barras sin procesar todos los archivos."""
        # Intentar leer de un archivo reciente
        sample_files = list(self.base_dir.glob('data/raw/*.tsv'))[:1]
        if not sample_files:
            sample_files = list(self.base_dir.glob('DATA_2023/*.tsv'))[:1]
        
        if not sample_files:
            return []
        
        try:
            # Leer suficientes filas para obtener todas las barras únicas
            df = pd.read_csv(sample_files[0], sep='\t', encoding='latin1', 
                           usecols=[6], nrows=500000)  # Solo columna de nombre, más filas
            col_name = df.columns[0]
            return sorted(df[col_name].dropna().unique().tolist())
        except Exception as e:
            print(f"Error leyendo barras: {e}")
            pass
        
        return []
