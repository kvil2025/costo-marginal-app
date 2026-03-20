#!/usr/bin/env python3
"""
Script para importar datos TSV a SQLite
Ejecutar una sola vez: python scripts/import_to_sqlite.py
"""
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Configuración
DB_PATH = Path("data/cmarg.db")
BASE_DIR = Path(".")

def create_database():
    """Crea la base de datos y la tabla con índices."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Eliminar tabla si existe
    cursor.execute("DROP TABLE IF EXISTS costo_marginal")
    
    # Crear tabla
    cursor.execute("""
        CREATE TABLE costo_marginal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            nombre_central TEXT,
            costo_marginal REAL
        )
    """)
    
    conn.commit()
    return conn

def create_indices(conn):
    """Crea índices para optimizar consultas."""
    cursor = conn.cursor()
    print("\n📊 Creando índices...")
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nombre ON costo_marginal(nombre_central)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON costo_marginal(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nombre_timestamp ON costo_marginal(nombre_central, timestamp)")
    
    conn.commit()
    print("✅ Índices creados")

def get_all_tsv_files():
    """Obtiene todos los archivos TSV disponibles."""
    files = []
    
    # Archivos en data/raw/
    files.extend(list(BASE_DIR.glob("data/raw/*.tsv")))
    
    # Archivos en DATA_20XX/
    for year in range(2015, 2026):
        year_dir = BASE_DIR / f"DATA_{year}"
        if year_dir.exists():
            files.extend(list(year_dir.glob("*.tsv")))
    
    return files

def import_file(filepath: Path, conn: sqlite3.Connection) -> int:
    """Importa un archivo TSV a la base de datos."""
    try:
        # Leer archivo
        df = pd.read_csv(filepath, sep='\t', encoding='latin1', decimal=',')
        
        # Determinar columnas según formato
        if 'nombre' in df.columns:
            name_col = 'nombre'
            cost_col = 'costo_en_dolares'
        elif 'nombre_central' in df.columns:
            name_col = 'nombre_central'
            cost_col = 'costo_marginal' if 'costo_marginal' in df.columns else 'costo_en_dolares'
        else:
            return 0
        
        # Preparar datos
        if 'fecha' in df.columns and 'hora' in df.columns:
            df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'].astype(int) - 1, unit='h')
        else:
            return 0
        
        # Seleccionar columnas necesarias
        df_clean = df[['timestamp', name_col, cost_col]].copy()
        df_clean.columns = ['timestamp', 'nombre_central', 'costo_marginal']
        
        # Convertir costo a numérico
        df_clean['costo_marginal'] = pd.to_numeric(df_clean['costo_marginal'], errors='coerce')
        
        # Convertir timestamp a string para SQLite
        df_clean['timestamp'] = df_clean['timestamp'].astype(str)
        
        # Eliminar nulos
        df_clean = df_clean.dropna()
        
        # Insertar en lotes
        df_clean.to_sql('costo_marginal', conn, if_exists='append', index=False)
        
        return len(df_clean)
        
    except Exception as e:
        print(f"\n⚠️ Error en {filepath.name}: {e}")
        return 0

def main():
    print("=" * 60)
    print("🔌 CMARG - Importador de Datos a SQLite")
    print("=" * 60)
    
    # Crear directorio si no existe
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Obtener archivos
    files = get_all_tsv_files()
    print(f"\n📁 Encontrados {len(files)} archivos TSV para importar")
    
    if not files:
        print("❌ No se encontraron archivos TSV")
        sys.exit(1)
    
    # Crear base de datos
    print("\n🗄️ Creando base de datos...")
    conn = create_database()
    
    # Importar archivos
    total_records = 0
    print("\n📥 Importando archivos...")
    
    for filepath in tqdm(files, desc="Procesando"):
        records = import_file(filepath, conn)
        total_records += records
    
    # Crear índices
    create_indices(conn)
    
    # Estadísticas finales
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT nombre_central) FROM costo_marginal")
    unique_bars = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM costo_marginal")
    date_range = cursor.fetchone()
    
    conn.close()
    
    # Mostrar resultados
    db_size = DB_PATH.stat().st_size / (1024 * 1024 * 1024)  # GB
    
    print("\n" + "=" * 60)
    print("✅ IMPORTACIÓN COMPLETADA")
    print("=" * 60)
    print(f"📊 Registros totales: {total_records:,}")
    print(f"🏭 Barras únicas: {unique_bars:,}")
    print(f"📅 Rango de fechas: {date_range[0][:10]} a {date_range[1][:10]}")
    print(f"💾 Tamaño de DB: {db_size:.2f} GB")
    print(f"📍 Ubicación: {DB_PATH.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
