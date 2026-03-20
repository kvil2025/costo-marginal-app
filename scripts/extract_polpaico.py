"""
Script para extraer todos los datos de Polpaico de los archivos TSV
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

base_dir = Path(".")
output_dir = Path("data/polpaico")
output_dir.mkdir(exist_ok=True)

# Buscar todos los archivos TSV
all_files = list(base_dir.glob("DATA_*/*.tsv"))
print(f"📂 Encontrados {len(all_files)} archivos TSV")

dfs = []
for file in tqdm(all_files, desc="Procesando archivos"):
    try:
        df = pd.read_csv(file, sep='\t', encoding='latin1', decimal=',')
        
        # Determinar columna de nombre
        if 'nombre' in df.columns:
            name_col = 'nombre'
        elif 'nombre_central' in df.columns:
            name_col = 'nombre_central'
        else:
            name_col = df.columns[-1]
        
        # Filtrar por Polpaico (case insensitive)
        df_filtered = df[df[name_col].str.contains('POLPAICO', case=False, na=False)]
        
        if not df_filtered.empty:
            dfs.append(df_filtered)
            print(f"  ✅ {file.name}: {len(df_filtered)} registros")
    except Exception as e:
        print(f"  ❌ Error en {file.name}: {e}")

if dfs:
    result = pd.concat(dfs, ignore_index=True)
    
    # Guardar resultados
    output_file = output_dir / "polpaico_all_data.csv"
    result.to_csv(output_file, index=False)
    print(f"\n✅ Datos exportados a: {output_file}")
    print(f"📊 Total registros: {len(result):,}")
    
    # Mostrar resumen
    print(f"\n📌 Barras encontradas:")
    name_col = 'nombre' if 'nombre' in result.columns else result.columns[-1]
    for name in result[name_col].unique():
        count = len(result[result[name_col] == name])
        print(f"   - {name}: {count:,} registros")
else:
    print("❌ No se encontraron datos de Polpaico")
