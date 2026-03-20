"""
Script para subir datos de barras troncales a BigQuery.
Procesa los TSV raw del CEN y los sube al dataset cmarg.

Uso:
    python scripts/upload_to_bigquery.py

Requisitos:
    pip install google-cloud-bigquery pandas tqdm
    gcloud auth application-default login
"""
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    from google.cloud import bigquery
except ImportError:
    print("❌ Instala google-cloud-bigquery: pip install google-cloud-bigquery")
    sys.exit(1)

# Configuración
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'geologgia-map')
DATASET_ID = os.environ.get('BQ_DATASET', 'cmarg')
TABLE_COSTOS = f'{PROJECT_ID}.{DATASET_ID}.costos_marginales'
TABLE_EXOGENOS = f'{PROJECT_ID}.{DATASET_ID}.datos_exogenos'

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent


def create_dataset_and_tables(client):
    """Crea el dataset y las tablas en BigQuery si no existen."""
    
    # Crear dataset
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset_ref.location = "US"
    dataset_ref.description = "CMARG Pro - Costos Marginales del SEN Chile"
    
    try:
        client.create_dataset(dataset_ref, exists_ok=True)
        print(f"✅ Dataset '{DATASET_ID}' listo")
    except Exception as e:
        print(f"⚠️ Error creando dataset: {e}")
        return False
    
    # Crear tabla de costos marginales (particionada por fecha)
    schema_costos = [
        bigquery.SchemaField("barra_mnemotecnico", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("barra_nombre", "STRING"),
        bigquery.SchemaField("fecha", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("hora", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("costo_usd", "FLOAT64"),
        bigquery.SchemaField("costo_clp", "FLOAT64"),
    ]
    
    table_ref = bigquery.Table(TABLE_COSTOS, schema=schema_costos)
    table_ref.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.MONTH,
        field="fecha"
    )
    table_ref.clustering_fields = ["barra_mnemotecnico"]
    table_ref.description = "Costos marginales horarios por barra del SEN"
    
    try:
        client.create_table(table_ref, exists_ok=True)
        print(f"✅ Tabla 'costos_marginales' lista (particionada por fecha)")
    except Exception as e:
        print(f"⚠️ Error creando tabla costos: {e}")
    
    # Crear tabla de datos exógenos
    schema_exogenos = [
        bigquery.SchemaField("fecha", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("usd_clp", "FLOAT64"),
        bigquery.SchemaField("eur_clp", "FLOAT64"),
        bigquery.SchemaField("cobre_usd", "FLOAT64"),
        bigquery.SchemaField("uf", "FLOAT64"),
        bigquery.SchemaField("oni_index", "FLOAT64"),
        bigquery.SchemaField("is_el_nino", "BOOLEAN"),
        bigquery.SchemaField("is_la_nina", "BOOLEAN"),
    ]
    
    table_ref_exo = bigquery.Table(TABLE_EXOGENOS, schema=schema_exogenos)
    table_ref_exo.description = "Datos exógenos: económicos y climáticos"
    
    try:
        client.create_table(table_ref_exo, exists_ok=True)
        print(f"✅ Tabla 'datos_exogenos' lista")
    except Exception as e:
        print(f"⚠️ Error creando tabla exógenos: {e}")
    
    return True


def process_tsv_file(filepath: Path) -> pd.DataFrame:
    """Procesa un archivo TSV raw del CEN."""
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
        except Exception as e:
            print(f"  ⚠️ Error leyendo {filepath.name}: {e}")
            return pd.DataFrame()
    
    # Identificar columnas
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'mnemotecnico' in col_lower and 'referencia' not in col_lower:
            col_mapping[col] = 'barra_mnemotecnico'
        elif 'nombre' in col_lower or 'barra' in col_lower:
            if 'mnemotecnico' not in col_lower and 'referencia' not in col_lower:
                col_mapping[col] = 'barra_nombre'
        elif 'fecha' in col_lower:
            col_mapping[col] = 'fecha'
        elif 'hora' in col_lower:
            col_mapping[col] = 'hora'
        elif 'usd' in col_lower or 'costo_en_dolar' in col_lower:
            col_mapping[col] = 'costo_usd'
        elif 'clp' in col_lower or 'costo_en_pesos' in col_lower:
            col_mapping[col] = 'costo_clp'
    
    if not col_mapping:
        return pd.DataFrame()
    
    df = df.rename(columns=col_mapping)
    
    # Asegurar columnas necesarias
    needed = ['barra_mnemotecnico', 'fecha', 'hora']
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    
    # Limpiar costos (convertir coma decimal a punto)
    for cost_col in ['costo_usd', 'costo_clp']:
        if cost_col in df.columns:
            if df[cost_col].dtype == object:
                df[cost_col] = df[cost_col].str.replace(',', '.').astype(float)
            df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
    
    # Crear timestamp
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce').dt.date
    df['hora'] = pd.to_numeric(df['hora'], errors='coerce').fillna(0).astype(int)
    df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'], unit='h')
    
    # Seleccionar columnas finales
    final_cols = ['barra_mnemotecnico', 'barra_nombre', 'fecha', 'hora', 'timestamp', 'costo_usd', 'costo_clp']
    for col in final_cols:
        if col not in df.columns:
            df[col] = None
    
    return df[final_cols].dropna(subset=['barra_mnemotecnico', 'fecha'])


def upload_barras_csv(client):
    """Sube los CSVs procesados de barras a BigQuery."""
    barras_dir = BASE_DIR / 'data' / 'barras'
    
    print("\n📊 Subiendo CSVs de barras procesadas...")
    
    for csv_file in sorted(barras_dir.glob('*.csv')):
        if csv_file.name in ['all_barras.csv', 'extraction_stats.csv']:
            continue
        
        print(f"  📁 {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Normalizar columnas
        if 'fecha' in df.columns and 'hora' in df.columns:
            df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'], unit='h')
            df['fecha'] = pd.to_datetime(df['fecha']).dt.date
        
        if 'costo_usd' in df.columns and 'costo_marginal' not in df.columns:
            pass  # ya tiene costo_usd
        elif 'costo_marginal' in df.columns:
            df['costo_usd'] = df['costo_marginal']
        
        # Asegurar columnas
        for col in ['barra_mnemotecnico', 'barra_nombre', 'costo_usd', 'costo_clp']:
            if col not in df.columns:
                df[col] = None
        
        final_cols = ['barra_mnemotecnico', 'barra_nombre', 'fecha', 'hora', 'timestamp', 'costo_usd', 'costo_clp']
        df_upload = df[[c for c in final_cols if c in df.columns]]
        
        # Subir a BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
        )
        
        try:
            job = client.load_table_from_dataframe(df_upload, TABLE_COSTOS, job_config=job_config)
            job.result()
            print(f"    ✅ {len(df_upload):,} filas subidas")
        except Exception as e:
            print(f"    ❌ Error: {e}")


def upload_raw_tsv(client):
    """Sube los archivos TSV raw del CEN a BigQuery."""
    print("\n📊 Procesando archivos TSV raw del CEN...")
    
    data_dirs = sorted(BASE_DIR.glob('DATA_20*/'))
    
    if not data_dirs:
        print("  ⚠️ No se encontraron directorios DATA_20XX/")
        return
    
    total_uploaded = 0
    
    for data_dir in tqdm(data_dirs, desc="Directorios"):
        tsv_files = sorted(data_dir.glob('*.tsv'))
        
        for tsv_file in tsv_files:
            df = process_tsv_file(tsv_file)
            
            if df.empty:
                continue
            
            # Subir en lotes de 50k
            batch_size = 50000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                job_config = bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                )
                
                try:
                    job = client.load_table_from_dataframe(batch, TABLE_COSTOS, job_config=job_config)
                    job.result()
                    total_uploaded += len(batch)
                except Exception as e:
                    print(f"  ❌ Error en {tsv_file.name}: {e}")
    
    print(f"\n✅ Total subido: {total_uploaded:,} filas de TSV raw")


def upload_exogenous(client):
    """Sube datos exógenos a BigQuery."""
    exo_path = BASE_DIR / 'data' / 'exogenous' / 'exogenous_data.csv'
    
    if not exo_path.exists():
        print("⚠️ No se encontró exogenous_data.csv")
        return
    
    print("\n🌡️ Subiendo datos exógenos...")
    df = pd.read_csv(exo_path)
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha']).dt.date
    
    # Renombrar columnas si es necesario
    rename_map = {
        'dolar_usd_clp': 'usd_clp',
        'euro_eur_clp': 'eur_clp',
        'cobre_usd_lb': 'cobre_usd',
        'oni': 'oni_index',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Seleccionar columnas válidas
    valid_cols = ['fecha', 'usd_clp', 'eur_clp', 'cobre_usd', 'uf', 'oni_index', 'is_el_nino', 'is_la_nina']
    df_upload = df[[c for c in valid_cols if c in df.columns]]
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Reemplazar datos
    )
    
    try:
        job = client.load_table_from_dataframe(df_upload, TABLE_EXOGENOS, job_config=job_config)
        job.result()
        print(f"✅ {len(df_upload):,} filas de datos exógenos subidas")
    except Exception as e:
        print(f"❌ Error subiendo exógenos: {e}")


def main():
    print("=" * 60)
    print("  CMARG Pro — Migración de datos a BigQuery")
    print("=" * 60)
    print(f"\n📎 Proyecto: {PROJECT_ID}")
    print(f"📎 Dataset:  {DATASET_ID}")
    print()
    
    # Conectar a BigQuery
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print("✅ Conectado a BigQuery")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print("\nEjecuta primero: gcloud auth application-default login")
        sys.exit(1)
    
    # 1. Crear dataset y tablas
    print("\n" + "=" * 40)
    print("FASE 1: Crear dataset y tablas")
    print("=" * 40)
    if not create_dataset_and_tables(client):
        sys.exit(1)
    
    # 2. Subir CSVs de barras procesadas
    print("\n" + "=" * 40)
    print("FASE 2: Subir barras procesadas (CSVs)")
    print("=" * 40)
    upload_barras_csv(client)
    
    # 3. Subir TSV raw (opcional, puede tomar tiempo)
    print("\n" + "=" * 40)
    print("FASE 3: Subir datos raw del CEN (TSVs)")
    print("=" * 40)
    
    response = input("\n¿Subir también los TSV raw (~6.6GB)? Esto puede tomar 30-60 min. [s/N]: ")
    if response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
        upload_raw_tsv(client)
    else:
        print("  ⏭️ Saltando TSV raw")
    
    # 4. Subir datos exógenos
    print("\n" + "=" * 40)
    print("FASE 4: Subir datos exógenos")
    print("=" * 40)
    upload_exogenous(client)
    
    # 5. Verificar
    print("\n" + "=" * 40)
    print("VERIFICACIÓN")
    print("=" * 40)
    
    for table_name in ['costos_marginales', 'datos_exogenos']:
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
        try:
            table = client.get_table(table_id)
            print(f"  ✅ {table_name}: {table.num_rows:,} filas, {table.num_bytes/1024/1024:.1f} MB")
        except Exception as e:
            print(f"  ⚠️ {table_name}: {e}")
    
    print("\n🎉 ¡Migración completada!")
    print(f"\nPuedes consultar los datos en:")
    print(f"  https://console.cloud.google.com/bigquery?project={PROJECT_ID}")


if __name__ == '__main__':
    main()
