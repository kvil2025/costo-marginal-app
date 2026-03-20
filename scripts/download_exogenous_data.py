"""
Descarga datos exógenos para enriquecer el modelo predictivo.
Fuentes: mindicador.cl (dólar, cobre), NOAA (ONI El Niño/La Niña)
"""
import requests
import pandas as pd
import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'exogenous')
os.makedirs(DATA_DIR, exist_ok=True)


def download_mindicador(indicator: str, years: range, name: str):
    """Descarga indicador económico de mindicador.cl"""
    all_data = []
    for year in years:
        url = f"https://mindicador.cl/api/{indicator}/{year}"
        try:
            r = requests.get(url, timeout=30)
            data = r.json()
            serie = data.get('serie', [])
            for item in serie:
                all_data.append({
                    'fecha': item['fecha'][:10],
                    name: item['valor']
                })
            print(f"  ✅ {indicator} {year}: {len(serie)} registros")
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            print(f"  ❌ {indicator} {year}: {e}")
    
    df = pd.DataFrame(all_data)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').drop_duplicates('fecha').reset_index(drop=True)
    return df


def download_oni_index():
    """Descarga ONI (Oceanic Niño Index) de NOAA."""
    print("\n📊 Descargando ONI Index (El Niño/La Niña)...")
    
    # ONI data from NOAA PSL - ERSSTv5
    url = "https://psl.noaa.gov/data/correlation/oni.data"
    try:
        r = requests.get(url, timeout=30)
        lines = r.text.strip().split('\n')
        
        # Parse the fixed-width format
        data = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    if year < 2019 or year > 2026:
                        continue
                    for i, month_name in enumerate(months):
                        val = float(parts[i + 1])
                        if val > -90:  # -99.9 = missing
                            date = pd.Timestamp(year=year, month=i+1, day=15)
                            data.append({'fecha': date, 'oni_index': val})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            print(f"  ✅ ONI: {len(df)} registros mensuales ({df['fecha'].min().year}-{df['fecha'].max().year})")
            # Interpolar a diario
            df = df.set_index('fecha').resample('D').interpolate('linear').reset_index()
            df.columns = ['fecha', 'oni_index']
        return df
    except Exception as e:
        print(f"  ❌ ONI: {e}")
        # Fallback: crear ONI aproximado basado en eventos conocidos
        return create_manual_oni()


def create_manual_oni():
    """Crea ONI basado en eventos históricos conocidos."""
    print("  ℹ️ Usando ONI manual basado en eventos conocidos")
    events = [
        # (start, end, avg_oni, event_name)
        ('2019-01-01', '2019-06-30', 0.8, 'Weak El Niño'),
        ('2019-07-01', '2019-12-31', 0.1, 'Neutral'),
        ('2020-01-01', '2020-03-31', 0.5, 'Weak El Niño'),
        ('2020-04-01', '2020-08-31', -0.2, 'Near Neutral'),
        ('2020-09-01', '2021-05-31', -1.0, 'La Niña'),
        ('2021-06-01', '2021-08-31', -0.3, 'Neutral'),
        ('2021-09-01', '2022-07-31', -1.0, 'La Niña'),
        ('2022-08-01', '2023-02-28', -0.8, 'La Niña'),
        ('2023-03-01', '2023-05-31', -0.1, 'Neutral'),
        ('2023-06-01', '2024-04-30', 1.5, 'Strong El Niño'),
        ('2024-05-01', '2024-08-31', 0.0, 'Neutral'),
        ('2024-09-01', '2025-03-31', -0.5, 'Weak La Niña'),
        ('2025-04-01', '2025-12-31', -0.3, 'Near Neutral'),
        ('2026-01-01', '2026-12-31', 0.0, 'Neutral'),
    ]
    
    data = []
    for start, end, oni, name in events:
        dates = pd.date_range(start, end, freq='D')
        for d in dates:
            data.append({'fecha': d, 'oni_index': oni})
    
    return pd.DataFrame(data)


def download_all():
    """Descarga todos los datos exógenos."""
    years = range(2019, 2027)
    
    # 1. Dólar observado
    print("💵 Descargando Dólar Observado (USD/CLP)...")
    dolar_df = download_mindicador('dolar', years, 'usd_clp')
    
    # 2. Libra de Cobre
    print("\n🥇 Descargando Libra de Cobre (USD)...")
    cobre_df = download_mindicador('libra_cobre', years, 'cobre_usd')
    
    # 3. Euro
    print("\n💶 Descargando Euro (EUR/CLP)...")
    euro_df = download_mindicador('euro', years, 'eur_clp')
    
    # 4. UF (inflación)
    print("\n📈 Descargando UF...")
    uf_df = download_mindicador('uf', years, 'uf_clp')
    
    # 5. ONI (El Niño/La Niña)
    oni_df = download_oni_index()
    
    # Combinar todos
    print("\n🔗 Combinando datos...")
    
    # Base: rango completo de fechas
    all_dates = pd.DataFrame({'fecha': pd.date_range('2019-01-01', '2025-12-31', freq='D')})
    
    # Merge todos
    combined = all_dates
    for df, name in [(dolar_df, 'dolar'), (cobre_df, 'cobre'), 
                      (euro_df, 'euro'), (uf_df, 'uf'), (oni_df, 'oni')]:
        if df is not None and len(df) > 0:
            combined = combined.merge(df, on='fecha', how='left')
            print(f"  ✅ {name}: {df.shape[1]-1} columnas, {len(df)} registros")
    
    # Interpolar valores faltantes (fines de semana, feriados)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].interpolate(method='linear')
    combined = combined.ffill().bfill()
    
    # Agregar features derivadas
    if 'usd_clp' in combined.columns:
        combined['usd_clp_change_7d'] = combined['usd_clp'].pct_change(7)
        combined['usd_clp_change_30d'] = combined['usd_clp'].pct_change(30)
        combined['usd_clp_rolling_30d'] = combined['usd_clp'].rolling(30, min_periods=1).mean()
    
    if 'cobre_usd' in combined.columns:
        combined['cobre_change_30d'] = combined['cobre_usd'].pct_change(30)
    
    if 'oni_index' in combined.columns:
        combined['is_el_nino'] = (combined['oni_index'] >= 0.5).astype(int)
        combined['is_la_nina'] = (combined['oni_index'] <= -0.5).astype(int)
        combined['oni_squared'] = combined['oni_index'] ** 2  # Captura efectos no-lineales
    
    # Guardar
    output_path = os.path.join(DATA_DIR, 'exogenous_data.csv')
    combined.to_csv(output_path, index=False)
    print(f"\n✅ Datos guardados en {output_path}")
    print(f"   Shape: {combined.shape}")
    print(f"   Columnas: {list(combined.columns)}")
    print(f"   Rango: {combined['fecha'].min()} a {combined['fecha'].max()}")
    
    return combined


if __name__ == '__main__':
    df = download_all()
    print("\n📊 Primeras filas:")
    print(df.head(10).to_string())
    print("\n📊 Últimas filas:")
    print(df.tail(5).to_string())
    print("\n📊 Estadísticas:")
    print(df.describe().to_string())
