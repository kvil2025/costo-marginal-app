"""
Script para extraer las 10 barras troncales principales del SEN
desde los archivos TSV crudos (2015-2025).

Genera un CSV limpio por barra + un CSV consolidado.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURACIÓN ===
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "barras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 Barras troncales principales del SEN (norte a sur)
BARRAS_PRINCIPALES = {
    "crucero_220":       "BA S/E CRUCERO 220KV BP1",
    "encuentro_220":     "BA S/E ENCUENTRO 220KV BP1",
    "diego_almagro_220": "BA S/E DIEGO DE ALMAGRO 220KV BP1-1",
    "cardones_500":      "BA S/E NUEVA CARDONES 500KV BP1",
    "pan_azucar_500":    "BA S/E NUEVA PAN DE AZUCAR 500KV BPA",
    "los_vilos_220":     "BA S/E LOS VILOS 220KV BP1",
    "quillota_220":      "BA S/E QUILLOTA 220KV BP1-1",
    "polpaico_500":      "BA S/E POLPAICO (TRANSELEC) 500KV BPA",
    "alto_jahuel_500":   "BA S/E ALTO JAHUEL 500KV BPA",
    "charrua_500":       "BA S/E CHARRUA 500KV BP1-1",
}

# Nombres inversos para búsqueda
BARRA_NAMES = set(BARRAS_PRINCIPALES.values())


def detect_format(filepath: Path) -> str:
    """Detecta el formato del TSV (antiguo, nuevo, o real-def)."""
    with open(filepath, 'r', encoding='latin1') as f:
        header = f.readline().strip()
    
    if 'ID_INFO' in header and 'BARRA_INFO' in header:
        return 'real-def'  # 2024-2025 con datos cada 15 min
    elif 'barra_mnemotecnico' in header:
        return 'nuevo'     # 2017-2023
    else:
        return 'antiguo'   # 2015-2016


def read_tsv_new(filepath: Path) -> pd.DataFrame:
    """Lee TSV formato nuevo (2017-2023)."""
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='latin1', 
                         low_memory=False)
        
        if 'nombre' not in df.columns:
            return pd.DataFrame()
        
        # Filtrar por barras de interés
        df_filtered = df[df['nombre'].isin(BARRA_NAMES)].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        # Normalizar columnas
        result = pd.DataFrame({
            'barra': df_filtered['nombre'],
            'fecha': df_filtered['fecha'],
            'hora': df_filtered['hora'].astype(int),
            'minuto': 0,
            'costo_usd': df_filtered['costo_en_dolares'].astype(str).str.replace(',', '.').astype(float),
            'costo_clp': df_filtered['costo_en_pesos'].astype(str).str.replace(',', '.').astype(float),
        })
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ Error leyendo {filepath.name}: {e}")
        return pd.DataFrame()


def read_tsv_realdef(filepath: Path) -> pd.DataFrame:
    """Lee TSV formato REAL-DEF (2024-2025, datos cada 15 min)."""
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='latin1',
                         low_memory=False)
        
        if 'BARRA_INFO' not in df.columns:
            return pd.DataFrame()
        
        # Filtrar por barras de interés
        df_filtered = df[df['BARRA_INFO'].isin(BARRA_NAMES)].copy()
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        # Normalizar columnas
        result = pd.DataFrame({
            'barra': df_filtered['BARRA_INFO'],
            'fecha': df_filtered['FECHA'],
            'hora': df_filtered['HRA'].astype(int),
            'minuto': df_filtered['MIN'].astype(int),
            'costo_usd': pd.to_numeric(df_filtered['CMg[USD/MWh]'], errors='coerce'),
            'costo_clp': pd.to_numeric(df_filtered['CMg[CLP/KWh]'], errors='coerce'),
        })
        
        return result
        
    except Exception as e:
        print(f"  ⚠️ Error leyendo {filepath.name}: {e}")
        return pd.DataFrame()


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega datos de 15 minutos a promedio horario."""
    if df.empty:
        return df
    
    # Si todos los minutos son 0, ya es horario
    if df['minuto'].nunique() == 1 and df['minuto'].iloc[0] == 0:
        return df
    
    # Agrupar por barra + fecha + hora → promedio
    agg = df.groupby(['barra', 'fecha', 'hora']).agg({
        'costo_usd': 'mean',
        'costo_clp': 'mean',
        'minuto': 'first'
    }).reset_index()
    agg['minuto'] = 0
    
    return agg


def create_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columna timestamp a partir de fecha + hora."""
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['timestamp'] = df['fecha'] + pd.to_timedelta(df['hora'], unit='h')
    return df


def main():
    print("=" * 60)
    print("🔌 CMARG - Extracción de 10 Barras Troncales del SEN")
    print("=" * 60)
    print()
    print("Barras a extraer:")
    for key, name in BARRAS_PRINCIPALES.items():
        print(f"  📍 {name}")
    print()
    
    # Encontrar todos los TSVs
    all_tsvs = sorted(BASE_DIR.glob("DATA_*/*.tsv"))
    # También data/raw
    all_tsvs += sorted(BASE_DIR.glob("data/raw/*.tsv"))
    
    print(f"📂 Encontrados {len(all_tsvs)} archivos TSV")
    print()
    
    all_dfs = []
    
    for tsv_file in tqdm(all_tsvs, desc="Procesando archivos"):
        fmt = detect_format(tsv_file)
        
        if fmt == 'real-def':
            df = read_tsv_realdef(tsv_file)
        elif fmt == 'nuevo':
            df = read_tsv_new(tsv_file)
        else:
            # Formato antiguo - intentar como nuevo
            df = read_tsv_new(tsv_file)
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("❌ No se encontraron datos. Verificar archivos TSV.")
        sys.exit(1)
    
    # Combinar todo
    print("\n🔄 Combinando datos...")
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total registros brutos: {len(combined):,}")
    
    # Agregar a horario (datos de 15 min → promedio horario)
    print("🔄 Agregando a granularidad horaria...")
    combined = aggregate_to_hourly(combined)
    print(f"  Total registros horarios: {len(combined):,}")
    
    # Crear timestamp
    combined = create_timestamp(combined)
    
    # Eliminar duplicados
    combined = combined.drop_duplicates(subset=['barra', 'timestamp'])
    combined = combined.sort_values(['barra', 'timestamp']).reset_index(drop=True)
    print(f"  Sin duplicados: {len(combined):,}")
    
    # === Guardar CSVs individuales por barra ===
    print("\n💾 Guardando CSVs por barra...")
    stats = []
    
    for key, barra_name in BARRAS_PRINCIPALES.items():
        df_barra = combined[combined['barra'] == barra_name].copy()
        
        if df_barra.empty:
            print(f"  ⚠️ {key}: Sin datos")
            stats.append({
                'barra_id': key, 'barra_nombre': barra_name,
                'registros': 0, 'fecha_inicio': None, 'fecha_fin': None
            })
            continue
        
        # Seleccionar columnas para CSV
        output_cols = ['timestamp', 'fecha', 'hora', 'costo_usd', 'costo_clp', 'barra']
        df_out = df_barra[output_cols].sort_values('timestamp').reset_index(drop=True)
        
        # Guardar
        out_file = OUTPUT_DIR / f"{key}.csv"
        df_out.to_csv(out_file, index=False)
        
        fecha_min = df_out['fecha'].min()
        fecha_max = df_out['fecha'].max()
        
        print(f"  ✅ {key}: {len(df_out):,} registros ({fecha_min} → {fecha_max})")
        
        stats.append({
            'barra_id': key,
            'barra_nombre': barra_name,
            'registros': len(df_out),
            'fecha_inicio': str(fecha_min),
            'fecha_fin': str(fecha_max),
            'costo_medio_usd': round(df_out['costo_usd'].mean(), 2),
            'costo_std_usd': round(df_out['costo_usd'].std(), 2),
        })
    
    # === Guardar CSV consolidado ===
    output_cols = ['timestamp', 'fecha', 'hora', 'costo_usd', 'costo_clp', 'barra']
    consolidated = combined[output_cols].sort_values(['barra', 'timestamp']).reset_index(drop=True)
    consolidated_file = OUTPUT_DIR / "all_barras.csv"
    consolidated.to_csv(consolidated_file, index=False)
    print(f"\n💾 Consolidado: {consolidated_file} ({len(consolidated):,} registros)")
    
    # === Resumen ===
    stats_df = pd.DataFrame(stats)
    stats_file = OUTPUT_DIR / "extraction_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE EXTRACCIÓN")
    print("=" * 60)
    for _, row in stats_df.iterrows():
        status = "✅" if row['registros'] > 0 else "❌"
        print(f"  {status} {row['barra_id']:25s} {row['registros']:>8,} registros")
    
    total = stats_df['registros'].sum()
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.csv")) / 1024 / 1024
    print(f"\n  Total: {total:,} registros | {total_mb:.1f} MB")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
