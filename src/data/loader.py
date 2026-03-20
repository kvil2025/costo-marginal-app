import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

class TSVLoader:
    """Clase para cargar archivos TSV de datos de costo marginal."""
    
    COLUMNS = [
        'codigo_central',
        'codigo_ubicacion',
        'fecha',
        'hora',
        'costo_marginal',
        'valor_secundario',
        'nombre_central'
    ]
    
    def __init__(self, data_dir: str):
        """
        Inicializa el cargador con el directorio de datos.
        
        Args:
            data_dir: Ruta al directorio que contiene los archivos TSV
        """
        self.data_dir = Path(data_dir)
        self.data = None
        
    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        """
        Carga un único archivo TSV de manera eficiente.
        
        Args:
            filepath: Ruta al archivo TSV
            
        Returns:
            DataFrame con los datos del archivo
        """
        df = pd.read_csv(
            filepath,
            sep='\t',
            decimal=',',
            header=None,
            names=self.COLUMNS,
            skiprows=1,  # Ignorar la fila de encabezado
            usecols=[0, 2, 3, 4, 6],  # Cargar solo columnas esenciales
            dtype={
                'codigo_central': 'category',
                'nombre_central': 'category',
                'hora': 'int8',
                'costo_marginal': 'float32'
            },
            encoding='latin1'
        )
        
        df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'] - 1, unit='h')
        df = df.drop(columns=['fecha', 'hora'])
        
        return df

    def load_all_files(self, pattern: str = '*.tsv') -> pd.DataFrame:
        """
        Carga todos los archivos que coincidan con el patrón.
        
        Args:
            pattern: Patrón para buscar archivos (ej: '*.tsv')
            
        Returns:
            DataFrame con todos los datos combinados
        """
        files = list(self.data_dir.glob(f'**/{pattern}'))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos con el patrón: {pattern} en {self.data_dir}")
            
        print(f"Encontrados {len(files)} archivos para procesar")
        
        dfs = [self.load_single_file(file) for file in tqdm(files, desc="Procesando archivos")]
                
        if not dfs:
            raise ValueError("No se pudo cargar ningún archivo")
            
        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        return self.data
