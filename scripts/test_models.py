"""
Test directo de todos los modelos predictivos.
Ejecutar: /tmp/cmarg_env/bin/python scripts/test_models.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
import traceback

# ============================================
# 1. CARGAR DATOS
# ============================================
print("=" * 60)
print("🔬 TEST DE MODELOS PREDICTIVOS - CMARG")
print("=" * 60)

barra = "los_vilos_220"
csv_path = f"data/barras/{barra}.csv"

print(f"\n📂 Cargando datos de {barra}...")
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Los modelos esperan 'costo_marginal', el CSV tiene 'costo_usd'
if 'costo_usd' in df.columns and 'costo_marginal' not in df.columns:
    df['costo_marginal'] = pd.to_numeric(df['costo_usd'], errors='coerce')

print(f"   ✅ {len(df):,} registros cargados")
print(f"   📅 Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"   💰 Promedio: ${df['costo_marginal'].mean():.2f}/MWh")
print(f"   📊 Columnas: {list(df.columns)}")

# ============================================
# 2. TEST XGBOOST (con fallback GradientBoosting)
# ============================================
print("\n" + "=" * 60)
print("🧠 TEST 1: XGBoost / GradientBoosting")
print("=" * 60)

try:
    from src.models.xgboost_predictor import XGBoostPredictor
    
    xgb_pred = XGBoostPredictor()
    
    t0 = time.time()
    train_result = xgb_pred.train(df)
    train_time = time.time() - t0
    
    print(f"\n   ⏱️ Tiempo de entrenamiento: {train_time:.1f}s")
    print(f"   📊 Resultado: {train_result.get('message', 'N/A')}")
    
    if train_result.get('success'):
        metrics = xgb_pred.get_model_metrics()
        print(f"   📈 MAE: ${metrics['mae']}/MWh")
        print(f"   📈 MAPE: {metrics['mape']}%")
        print(f"   📈 RMSE: ${metrics['rmse']}/MWh")
        
        fi = xgb_pred.get_feature_importance()
        if fi is not None:
            print(f"   🏆 Top 5 features:")
            for _, row in fi.head(5).iterrows():
                print(f"      - {row['feature']}: {row['importance']:.4f}")
        
        # Predicción
        t0 = time.time()
        result_df, pred_metrics = xgb_pred.predict(years_ahead=5)
        pred_time = time.time() - t0
        
        print(f"\n   ⏱️ Tiempo de predicción: {pred_time:.1f}s")
        
        if pred_metrics.get('success'):
            print(f"   🔮 Rango: {pred_metrics['prediction_start']} → {pred_metrics['prediction_end']}")
            print(f"   💰 Promedio predicho: ${pred_metrics['avg_predicted_value']:.2f}/MWh")
            print(f"   ⬇️ Mínimo predicho: ${pred_metrics['min_predicted']:.2f}/MWh")
            print(f"   ⬆️ Máximo predicho: ${pred_metrics['max_predicted']:.2f}/MWh")
            
            pred_only = result_df[result_df['tipo'] == 'Predicción']
            hist_only = result_df[result_df['tipo'] == 'Histórico']
            print(f"   📊 Datos: {len(hist_only)} hist + {len(pred_only)} pred = {len(result_df)} total")
            
            result_df.to_csv('data/prediction_xgboost_losvilos.csv', index=False)
            print(f"   💾 Guardado en data/prediction_xgboost_losvilos.csv")
            print(f"\n   ✅ XGBOOST/GRADIENTBOOSTING: ¡FUNCIONA!")
        else:
            print(f"   ❌ Error en predicción: {pred_metrics}")
    else:
        print(f"   ❌ Entrenamiento falló: {train_result}")
        
except Exception as e:
    print(f"   ❌ ERROR FATAL: {e}")
    traceback.print_exc()

# ============================================
# 3. TEST PROPHET
# ============================================
print("\n" + "=" * 60)
print("🔮 TEST 2: Prophet")
print("=" * 60)

try:
    from src.models.predictor import PricePredictor
    
    prophet_pred = PricePredictor()
    
    t0 = time.time()
    train_result = prophet_pred.train(df)
    train_time = time.time() - t0
    
    print(f"\n   ⏱️ Tiempo de entrenamiento: {train_time:.1f}s")
    print(f"   📊 Resultado: {train_result.get('message', 'N/A')}")
    
    if train_result.get('success'):
        t0 = time.time()
        result_df, pred_metrics = prophet_pred.predict(years_ahead=5)
        pred_time = time.time() - t0
        
        print(f"   ⏱️ Tiempo de predicción: {pred_time:.1f}s")
        
        if pred_metrics.get('success'):
            print(f"   🔮 Rango: {pred_metrics['prediction_start']} → {pred_metrics['prediction_end']}")
            print(f"   💰 Promedio predicho: ${pred_metrics['avg_predicted_value']:.2f}/MWh")
            
            result_df.to_csv('data/prediction_prophet_losvilos.csv', index=False)
            print(f"   💾 Guardado en data/prediction_prophet_losvilos.csv")
            print(f"\n   ✅ PROPHET: ¡FUNCIONA!")
        else:
            print(f"   ❌ Error: {pred_metrics}")
    else:
        print(f"   ❌ Entrenamiento falló: {train_result}")
        
except Exception as e:
    print(f"   ❌ ERROR FATAL: {e}")
    traceback.print_exc()

# ============================================
# 4. TEST ARIMA
# ============================================
print("\n" + "=" * 60)
print("📉 TEST 3: ARIMA")
print("=" * 60)

try:
    from src.models.arima_predictor import ARIMAPredictor
    
    arima_pred = ARIMAPredictor()
    
    t0 = time.time()
    train_result = arima_pred.train(df)
    train_time = time.time() - t0
    
    print(f"\n   ⏱️ Tiempo de entrenamiento: {train_time:.1f}s")
    print(f"   📊 Resultado: {train_result.get('message', 'N/A')}")
    
    if train_result.get('success'):
        t0 = time.time()
        result_df, pred_metrics = arima_pred.predict(years_ahead=5)
        pred_time = time.time() - t0
        
        print(f"   ⏱️ Tiempo de predicción: {pred_time:.1f}s")
        
        if pred_metrics.get('success'):
            print(f"   🔮 Rango: {pred_metrics['prediction_start']} → {pred_metrics['prediction_end']}")
            print(f"   💰 Promedio predicho: ${pred_metrics['avg_predicted_value']:.2f}/MWh")
            
            result_df.to_csv('data/prediction_arima_losvilos.csv', index=False)
            print(f"   💾 Guardado")
            print(f"\n   ✅ ARIMA: ¡FUNCIONA!")
        else:
            print(f"   ❌ Error: {pred_metrics}")
    else:
        print(f"   ❌ Entrenamiento falló: {train_result}")
        
except Exception as e:
    print(f"   ❌ ERROR FATAL: {e}")
    traceback.print_exc()

# ============================================
print("\n" + "=" * 60)
print("✅ TESTS COMPLETADOS")
print("=" * 60)
