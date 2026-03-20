"""
CMARG Dashboard Pro - Análisis Predictivo de Costo Marginal
Sistema Eléctrico Nacional de Chile
10 Barras Troncales con Predicción IA (Prophet, ARIMA, XGBoost)
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos propios
from src.models.predictor import PricePredictor
from src.models.arima_predictor import ARIMAPredictor
from src.models.xgboost_predictor import XGBoostPredictor
from src.models.xgboost_predictor_v2 import XGBoostPredictorV2

# Diccionario de predictores disponibles
PREDICTORS = {
    'xgboost_v2': XGBoostPredictorV2,
    'prophet': PricePredictor,
    'arima': ARIMAPredictor,
    'xgboost': XGBoostPredictor,
}

# --- 10 Barras troncales del SEN ---
BARRAS_DIR = Path("data/barras")

BARRAS_INFO = {
    "crucero_220":       {"nombre": "BA S/E CRUCERO 220KV BP1",              "zona": "Norte Grande",   "icon": "🔴", "color": "#ef4444"},
    "encuentro_220":     {"nombre": "BA S/E ENCUENTRO 220KV BP1",            "zona": "Norte Grande",   "icon": "🟠", "color": "#f97316"},
    "diego_almagro_220": {"nombre": "BA S/E DIEGO DE ALMAGRO 220KV BP1-1",   "zona": "Norte",          "icon": "🟡", "color": "#eab308"},
    "cardones_500":      {"nombre": "BA S/E NUEVA CARDONES 500KV BP1",       "zona": "Norte",          "icon": "🟢", "color": "#22c55e"},
    "pan_azucar_500":    {"nombre": "BA S/E NUEVA PAN DE AZUCAR 500KV BPA",  "zona": "Norte-Centro",   "icon": "🔵", "color": "#3b82f6"},
    "los_vilos_220":     {"nombre": "BA S/E LOS VILOS 220KV BP1",            "zona": "Centro-Norte",   "icon": "⭐", "color": "#a855f7"},
    "quillota_220":      {"nombre": "BA S/E QUILLOTA 220KV BP1-1",           "zona": "Centro",         "icon": "🟣", "color": "#8b5cf6"},
    "polpaico_500":      {"nombre": "BA S/E POLPAICO (TRANSELEC) 500KV BPA", "zona": "Centro (Stgo)",  "icon": "💎", "color": "#06b6d4"},
    "alto_jahuel_500":   {"nombre": "BA S/E ALTO JAHUEL 500KV BPA",          "zona": "Centro (Stgo)",  "icon": "🔷", "color": "#0ea5e9"},
    "charrua_500":       {"nombre": "BA S/E CHARRUA 500KV BP1-1",            "zona": "Sur",            "icon": "🟤", "color": "#78716c"},
}

# --- Configuración de estilos ---
COLORS = {
    'background': '#0f0f1a',
    'card': '#1a1a2e',
    'card_border': '#2d2d44',
    'primary': '#6366f1',
    'secondary': '#22d3ee',
    'accent': '#f472b6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'text': '#e2e8f0',
    'text_muted': '#94a3b8',
    'gradient_start': '#6366f1',
    'gradient_end': '#8b5cf6'
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '16px',
    'border': f"1px solid {COLORS['card_border']}",
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
}

# --- Cargar datos de barras ---
print("🔌 Iniciando CMARG Dashboard Pro - 10 Barras Troncales del SEN...")

bar_options = []
loaded_bars = {}

for bar_id, info in BARRAS_INFO.items():
    csv_file = BARRAS_DIR / f"{bar_id}.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        loaded_bars[bar_id] = df
        bar_options.append({
            'label': f"{info['icon']} {info['nombre']} ({info['zona']})",
            'value': bar_id
        })
        print(f"  ✅ {info['icon']} {bar_id}: {len(df):,} registros")
    else:
        print(f"  ⚠️ {bar_id}: archivo no encontrado")

if not bar_options:
    print("⚠️ No se encontraron datos de barras. Ejecutar: python scripts/extract_bars.py")
    # Fallback a Polpaico data antiguo
    old_file = Path("data/polpaico/polpaico_500kv.csv")
    if old_file.exists():
        df = pd.read_csv(old_file)
        df['costo_usd'] = df['costo_en_dolares'].astype(str).str.replace(',', '.').astype(float)
        df['timestamp'] = pd.to_datetime(df['fecha']) + pd.to_timedelta(df['hora'].astype(int) - 1, unit='h')
        loaded_bars['polpaico_500'] = df
        bar_options.append({
            'label': '💎 BA S/E POLPAICO (TRANSELEC) 500KV BPA (Centro)',
            'value': 'polpaico_500'
        })
        print(f"  ✅ Fallback: polpaico_500 con {len(df):,} registros")

print(f"\n✅ {len(bar_options)} barras disponibles")

# --- Inicialización de la aplicación Dash ---
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    title="CMARG Pro - Análisis Predictivo del SEN",
    meta_tags=[{
        'name': 'viewport',
        'content': 'width=device-width, initial-scale=1.0'
    }, {
        'name': 'description',
        'content': 'Dashboard de análisis predictivo de costo marginal del Sistema Eléctrico Nacional de Chile. Modelos Prophet, ARIMA y XGBoost.'
    }],
    suppress_callback_exceptions=True
)
app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# --- CSS personalizado ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { font-family: 'Inter', sans-serif; }
            body {
                background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
                min-height: 100vh;
            }
            .dash-dropdown .Select-control {
                background-color: #1a1a2e !important;
                border-color: #2d2d44 !important;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: #1a1a2e !important;
            }
            .kpi-card {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
            }
            .kpi-value {
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #6366f1, #22d3ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .kpi-label {
                color: #94a3b8;
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .section-title {
                color: #e2e8f0;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .section-title::before {
                content: '';
                width: 4px;
                height: 24px;
                background: linear-gradient(180deg, #6366f1, #22d3ee);
                border-radius: 2px;
            }
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
            }
            .status-ready { background: rgba(16, 185, 129, 0.2); color: #10b981; }
            .btn-gradient {
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .btn-gradient:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
            }
            .metric-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-right: 8px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Layout principal ---
app.layout = dbc.Container(fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '30px'}, children=[
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("⚡ CMARG Pro", style={
                    'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
                    '-webkit-background-clip': 'text',
                    '-webkit-text-fill-color': 'transparent',
                    'fontSize': '2.5rem',
                    'fontWeight': '700',
                    'marginBottom': '5px'
                }),
                html.P("Análisis Predictivo del SEN • 10 Barras Troncales • Chile", 
                       style={'color': COLORS['text_muted'], 'fontSize': '1rem'})
            ])
        ], md=8),
        dbc.Col([
            html.Div(id='status-indicator', children=[
                html.Span(f"{len(bar_options)} barras cargadas", className="status-badge status-ready")
            ], style={'textAlign': 'right', 'paddingTop': '15px'})
        ], md=4)
    ], style={'marginBottom': '30px'}),
    
    # Selector de Barra
    dbc.Row([
        dbc.Col([
            html.Div(style=CARD_STYLE, children=[
                html.H5("🎯 Selecciona una Barra Troncal", className="section-title"),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='bar-selector',
                            options=bar_options,
                            placeholder="Selecciona una barra del sistema eléctrico...",
                            style={'backgroundColor': COLORS['card']},
                            className="mb-3"
                        )
                    ], md=6),
                    dbc.Col([
                        html.Button("📥 Cargar y Analizar", id='load-btn', className="btn-gradient", 
                                   style={'width': '100%'})
                    ], md=2),
                    dbc.Col([
                        html.Button("🔮 Generar Predicción", id='predict-btn', className="btn-gradient", 
                                   style={'width': '100%', 'background': f'linear-gradient(135deg, {COLORS["secondary"]}, {COLORS["success"]})'},
                                   disabled=True)
                    ], md=2),
                    dbc.Col([
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': '🧠 XGBoost V2 - Predictivo Real', 'value': 'xgboost_v2'},
                                {'label': '🚀 XGBoost - ML Avanzado', 'value': 'xgboost'},
                                {'label': '📈 Prophet - Estacional', 'value': 'prophet'},
                                {'label': '📊 ARIMA - Estadístico', 'value': 'arima'},
                            ],
                            value='xgboost_v2',
                            clearable=False,
                            style={'backgroundColor': COLORS['card']}
                        )
                    ], md=2)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("📅 Años de Predicción:", style={'color': COLORS['text'], 'marginRight': '10px'}),
                        dcc.Slider(
                            id='prediction-years-slider',
                            min=1, max=10, step=1, value=5,
                            marks={i: {'label': str(i), 'style': {'color': COLORS['text_muted']}} 
                                   for i in [1, 2, 3, 5, 7, 10]},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        )
                    ], md=4, style={'marginTop': '10px'}),
                    dbc.Col([
                        html.Label("📊 Escenario Crecimiento PIB Chile:", style={'color': COLORS['text'], 'marginRight': '10px'}),
                        dcc.Slider(
                            id='gdp-growth-slider',
                            min=0, max=5, step=1, value=0,
                            marks={
                                0: {'label': 'Sin ajuste', 'style': {'color': COLORS['text_muted'], 'fontSize': '10px'}},
                                1: {'label': '1%', 'style': {'color': '#22d3ee'}},
                                2: {'label': '2%', 'style': {'color': '#10b981'}},
                                3: {'label': '3%', 'style': {'color': '#f59e0b'}},
                                4: {'label': '4%', 'style': {'color': '#f97316'}},
                                5: {'label': '5%', 'style': {'color': '#ef4444'}}
                            },
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        )
                    ], md=4, style={'marginTop': '10px'}),
                    dbc.Col([
                        html.Div(id='model-metrics', style={'marginTop': '10px'})
                    ], md=4)
                ])
            ])
        ])
    ]),
    
    # KPIs
    dbc.Row(id='kpi-row', style={'marginBottom': '20px'}, children=[
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-years"),
            html.Div("Años de Datos", className="kpi-label")
        ])], md=2),
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-records"),
            html.Div("Registros", className="kpi-label")
        ])], md=2),
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-avg"),
            html.Div("Promedio USD", className="kpi-label")
        ])], md=2),
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-min"),
            html.Div("Mínimo USD", className="kpi-label")
        ])], md=2),
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-max"),
            html.Div("Máximo USD", className="kpi-label")
        ])], md=2),
        dbc.Col([html.Div(className="kpi-card", children=[
            html.Div("--", className="kpi-value", id="kpi-volatility"),
            html.Div("Volatilidad %", className="kpi-label")
        ])], md=2)
    ]),
    
    # Status message
    dbc.Row([
        dbc.Col([
            html.Div(id='status-message', style={
                'backgroundColor': 'rgba(99, 102, 241, 0.1)',
                'border': f'1px solid {COLORS["primary"]}',
                'borderRadius': '8px',
                'padding': '15px',
                'color': COLORS['text'],
                'marginBottom': '20px'
            }, children=[
                "👋 Selecciona una de las 10 barras troncales del SEN para comenzar el análisis predictivo."
            ])
        ])
    ]),
    
    # Gráficos principales
    dcc.Loading(id="loading-main", type="circle", color=COLORS['primary'], children=[
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("📈 Evolución Histórica del Costo Marginal", className="section-title"),
                    dcc.Graph(id='historical-plot', style={'height': '400px'},
                             config={'displayModeBar': True, 'displaylogo': False})
                ])
            ], md=8),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("📊 Distribución por Año", className="section-title"),
                    dcc.Graph(id='yearly-box-plot', style={'height': '400px'},
                             config={'displayModeBar': False})
                ])
            ], md=4)
        ]),
        
        # Predicción
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5(id='prediction-title', children="🔮 Predicción con IA", className="section-title"),
                    dcc.Graph(id='prediction-plot', style={'height': '450px'},
                             config={'displayModeBar': True, 'displaylogo': False})
                ])
            ], md=8),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("🏆 Importancia de Features", className="section-title"),
                    dcc.Graph(id='feature-importance-plot', style={'height': '450px'},
                             config={'displayModeBar': False})
                ])
            ], md=4)
        ]),
        
        # Estacionalidad + Comparación
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("🌊 Patrón Estacional", className="section-title"),
                    dcc.Graph(id='seasonality-plot', style={'height': '300px'},
                             config={'displayModeBar': False})
                ])
            ], md=6),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("📅 Patrón Semanal", className="section-title"),
                    dcc.Graph(id='weekly-plot', style={'height': '300px'},
                             config={'displayModeBar': False})
                ])
            ], md=6)
        ])
    ]),
    
    # Stores
    dcc.Store(id='stored-data'),
    dcc.Store(id='prediction-data')
])


# --- Callbacks ---
@app.callback(
    [Output('stored-data', 'data'),
     Output('status-message', 'children'),
     Output('predict-btn', 'disabled'),
     Output('kpi-years', 'children'),
     Output('kpi-records', 'children'),
     Output('kpi-avg', 'children'),
     Output('kpi-min', 'children'),
     Output('kpi-max', 'children'),
     Output('kpi-volatility', 'children'),
     Output('historical-plot', 'figure'),
     Output('yearly-box-plot', 'figure')],
    [Input('load-btn', 'n_clicks')],
    [State('bar-selector', 'value')],
    prevent_initial_call=True
)
def load_bar_data(n_clicks, selected_bar):
    if not selected_bar:
        return (None, "⚠️ Selecciona una barra primero.", True,
                "--", "--", "--", "--", "--", "--",
                create_empty_figure("Selecciona una barra"),
                create_empty_figure("Selecciona una barra"))
    
    if selected_bar not in loaded_bars:
        return (None, f"❌ Datos no disponibles para: {selected_bar}", True,
                "--", "--", "--", "--", "--", "--",
                create_empty_figure("Sin datos"),
                create_empty_figure("Sin datos"))
    
    df = loaded_bars[selected_bar].copy()
    
    # Determinar columna de costo
    cost_col = 'costo_usd' if 'costo_usd' in df.columns else 'costo_marginal'
    if cost_col == 'costo_marginal':
        df['costo_marginal'] = pd.to_numeric(df['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
    else:
        df['costo_marginal'] = pd.to_numeric(df[cost_col], errors='coerce')
    
    info = BARRAS_INFO.get(selected_bar, {})
    bar_color = info.get('color', COLORS['primary'])
    
    # Calcular KPIs
    years = (df['timestamp'].max() - df['timestamp'].min()).days / 365
    records = len(df)
    avg_price = df['costo_marginal'].mean()
    min_price = df['costo_marginal'].min()
    max_price = df['costo_marginal'].max()
    volatility = (df['costo_marginal'].std() / max(avg_price, 0.01)) * 100
    
    # Gráfico histórico (promedio diario)
    df_daily = df.set_index('timestamp').resample('D')['costo_marginal'].mean().reset_index()
    
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x=df_daily['timestamp'], y=df_daily['costo_marginal'],
        mode='lines', name='CMg Diario',
        line=dict(color=bar_color, width=1.5),
        fill='tozeroy', fillcolor=f'rgba({int(bar_color[1:3],16)},{int(bar_color[3:5],16)},{int(bar_color[5:7],16)},0.1)'
    ))
    
    df_daily['ma30'] = df_daily['costo_marginal'].rolling(window=30).mean()
    hist_fig.add_trace(go.Scatter(
        x=df_daily['timestamp'], y=df_daily['ma30'],
        mode='lines', name='Media Móvil 30d',
        line=dict(color=COLORS['accent'], width=2)
    ))
    
    hist_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh')
    )
    
    # Box plot por año
    df['year'] = df['timestamp'].dt.year
    box_fig = px.box(df, x='year', y='costo_marginal',
                     color_discrete_sequence=[bar_color])
    box_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Año'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh'),
        showlegend=False
    )
    
    # Guardar datos para predicción
    stored = df.to_json(date_format='iso', orient='split')
    
    zona = info.get('zona', '')
    message = f"✅ {info.get('icon', '')} {info.get('nombre', selected_bar)} ({zona}) | {years:.1f} años | {records:,} registros"
    
    return (stored, message, False,
            f"{years:.1f}", f"{records:,}", f"${avg_price:.1f}",
            f"${min_price:.1f}", f"${max_price:.1f}", f"{volatility:.1f}%",
            hist_fig, box_fig)


@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('feature-importance-plot', 'figure'),
     Output('seasonality-plot', 'figure'),
     Output('weekly-plot', 'figure'),
     Output('status-message', 'children', allow_duplicate=True),
     Output('prediction-title', 'children'),
     Output('model-metrics', 'children')],
    [Input('predict-btn', 'n_clicks'),
     Input('gdp-growth-slider', 'value')],
    [State('stored-data', 'data'),
     State('prediction-years-slider', 'value'),
     State('model-selector', 'value'),
     State('prediction-data', 'data')],
    prevent_initial_call=True
)
def generate_prediction(n_clicks, gdp_growth, stored_data, years_ahead, model_type, existing_prediction):
    empty_result = (
        create_empty_figure("Sin datos"),
        create_empty_figure("Sin datos"),
        create_empty_figure("Sin datos"),
        create_empty_figure("Sin datos"),
        "⚠️ Primero carga los datos históricos",
        "🔮 Predicción con IA", ""
    )
    
    if not stored_data:
        return empty_result
    
    # GDP growth elasticity: ~0.8 (1% GDP → 0.8% electricity demand → price impact)
    gdp_growth_pct = gdp_growth if gdp_growth else 0
    
    # Recuperar datos
    df = pd.read_json(stored_data, orient='split')
    
    # Crear instancia del predictor
    model_names = {'prophet': 'Prophet', 'arima': 'ARIMA', 'xgboost': 'XGBoost', 'xgboost_v2': 'XGBoost V2'}
    predictor_instance = PREDICTORS[model_type]()
    
    # Entrenar modelo
    train_result = predictor_instance.train(df)
    
    if not train_result['success']:
        return (
            create_empty_figure(train_result['message']),
            create_empty_figure("Error"),
            create_empty_figure("Error"),
            create_empty_figure("Error"),
            f"❌ {train_result['message']}",
            f"🔮 {model_names[model_type]}", ""
        )
    
    # Generar predicción
    forecast, metrics = predictor_instance.predict(years_ahead=years_ahead)
    
    # --- Gráfico de predicción ---
    pred_fig = go.Figure()
    
    hist_data = forecast[forecast['tipo'] == 'Histórico']
    pred_fig.add_trace(go.Scatter(
        x=hist_data['ds'], y=hist_data['yhat'],
        mode='lines', name='Histórico',
        line=dict(color=COLORS['primary'], width=1.5)
    ))
    
    pred_data = forecast[forecast['tipo'] == 'Predicción'].copy()
    
    # Apply GDP growth adjustment if selected
    if gdp_growth_pct > 0:
        # Calculate cumulative growth factor based on time from prediction start
        elasticity = 0.8  # PIB-Electricidad elasticity for Chile
        annual_impact = gdp_growth_pct * elasticity / 100  # e.g., 3% GDP → 2.4% demand → price impact
        pred_start = pred_data['ds'].min()
        days_from_start = (pred_data['ds'] - pred_start).dt.days
        # Compounding growth factor
        growth_factor = (1 + annual_impact) ** (days_from_start / 365.25)
        
        # Show base prediction as dashed/faded
        pred_fig.add_trace(go.Scatter(
            x=pred_data['ds'], y=pred_data['yhat'],
            mode='lines', name='Predicción Base',
            line=dict(color='rgba(16, 185, 129, 0.3)', width=1, dash='dot'),
            showlegend=True
        ))
        
        # Show GDP-adjusted prediction as main line
        pred_data_adj = pred_data.copy()
        pred_data_adj['yhat'] = pred_data['yhat'] * growth_factor
        pred_data_adj['yhat_upper'] = pred_data['yhat_upper'] * growth_factor
        pred_data_adj['yhat_lower'] = pred_data['yhat_lower'] * growth_factor
        
        gdp_colors = {1: '#22d3ee', 2: '#10b981', 3: '#f59e0b', 4: '#f97316', 5: '#ef4444'}
        gdp_color = gdp_colors.get(gdp_growth_pct, COLORS['success'])
        
        pred_fig.add_trace(go.Scatter(
            x=pred_data_adj['ds'], y=pred_data_adj['yhat'],
            mode='lines', name=f'PIB +{gdp_growth_pct}% (ajustado)',
            line=dict(color=gdp_color, width=2.5)
        ))
        
        pred_fig.add_trace(go.Scatter(
            x=pd.concat([pred_data_adj['ds'], pred_data_adj['ds'][::-1]]),
            y=pd.concat([pred_data_adj['yhat_upper'], pred_data_adj['yhat_lower'][::-1]]),
            fill='toself', fillcolor=f'rgba({int(gdp_color[1:3], 16)}, {int(gdp_color[3:5], 16)}, {int(gdp_color[5:7], 16)}, 0.1)',
            line=dict(color='rgba(0,0,0,0)'), name=f'Intervalo PIB +{gdp_growth_pct}%'
        ))
        
        # Update pred_data for metrics calculation
        avg_adjusted = pred_data_adj['yhat'].mean()
    else:
        pred_fig.add_trace(go.Scatter(
            x=pred_data['ds'], y=pred_data['yhat'],
            mode='lines', name='Predicción',
            line=dict(color=COLORS['success'], width=2)
        ))
        
        pred_fig.add_trace(go.Scatter(
            x=pd.concat([pred_data['ds'], pred_data['ds'][::-1]]),
            y=pd.concat([pred_data['yhat_upper'], pred_data['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Intervalo 95%'
        ))
    
    pred_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Fecha'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh')
    )
    
    # --- Feature Importance (solo XGBoost) ---
    fi_fig = create_empty_figure(f"Feature importance no disponible para {model_names[model_type]}")
    
    if model_type == 'xgboost' and hasattr(predictor_instance, 'get_feature_importance'):
        fi = predictor_instance.get_feature_importance()
        if fi is not None and len(fi) > 0:
            top_fi = fi.head(15)
            fi_fig = go.Figure()
            fi_fig.add_trace(go.Bar(
                x=top_fi['importance'][::-1],
                y=top_fi['feature'][::-1],
                orientation='h',
                marker_color=COLORS['secondary']
            ))
            fi_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Importancia'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
    
    # --- Estacionalidad (solo Prophet) ---
    seasonal = predictor_instance.get_seasonality_components() if hasattr(predictor_instance, 'get_seasonality_components') else None
    
    if seasonal is not None:
        seasonal['month'] = seasonal['ds'].dt.month
        monthly_pattern = seasonal.groupby('month')['yearly'].mean().reset_index()
        
        season_fig = go.Figure()
        season_fig.add_trace(go.Bar(
            x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
            y=monthly_pattern['yearly'],
            marker_color=[COLORS['primary'] if v >= 0 else COLORS['accent'] for v in monthly_pattern['yearly']]
        ))
        season_fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Efecto estacional')
        )
        
        seasonal['weekday'] = seasonal['ds'].dt.dayofweek
        weekly_pattern = seasonal.groupby('weekday')['weekly'].mean().reset_index()
        
        weekly_fig = go.Figure()
        weekly_fig.add_trace(go.Bar(
            x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            y=weekly_pattern['weekly'],
            marker_color=COLORS['secondary']
        ))
        weekly_fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Efecto semanal')
        )
    else:
        # Para ARIMA y XGBoost, mostrar análisis estadístico
        df_ts = df.copy()
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
        df_ts['month'] = df_ts['timestamp'].dt.month
        df_ts['weekday'] = df_ts['timestamp'].dt.dayofweek
        
        cost_col = 'costo_marginal' if 'costo_marginal' in df_ts.columns else 'costo_usd'
        
        monthly = df_ts.groupby('month')[cost_col].mean()
        season_fig = go.Figure()
        season_fig.add_trace(go.Bar(
            x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
            y=monthly.values,
            marker_color=[COLORS['primary'] if v >= monthly.mean() else COLORS['accent'] for v in monthly.values]
        ))
        season_fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Promedio USD/MWh')
        )
        
        weekly = df_ts.groupby('weekday')[cost_col].mean()
        weekly_fig = go.Figure()
        weekly_fig.add_trace(go.Bar(
            x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            y=weekly.values,
            marker_color=COLORS['secondary']
        ))
        weekly_fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Promedio USD/MWh')
        )
    
    # --- Métricas del modelo ---
    metrics_div = ""
    if model_type == 'xgboost' and hasattr(predictor_instance, 'get_model_metrics'):
        m = predictor_instance.get_model_metrics()
        if m:
            metrics_div = html.Div([
                html.Span(f"MAE: ${m['mae']}", className="metric-badge", 
                         style={'background': 'rgba(16, 185, 129, 0.2)', 'color': '#10b981'}),
                html.Span(f"MAPE: {m['mape']}%", className="metric-badge",
                         style={'background': 'rgba(99, 102, 241, 0.2)', 'color': '#6366f1'}),
                html.Span(f"RMSE: ${m['rmse']}", className="metric-badge",
                         style={'background': 'rgba(34, 211, 238, 0.2)', 'color': '#22d3ee'}),
            ])
    
    gdp_label = f" | PIB +{gdp_growth_pct}%" if gdp_growth_pct > 0 else ""
    avg_val = avg_adjusted if gdp_growth_pct > 0 else metrics.get('avg_predicted_value', 0)
    message = f"🔮 {model_names[model_type]}: {metrics.get('prediction_start', '')} → {metrics.get('prediction_end', '')} ({years_ahead} años){gdp_label} | Promedio: ${avg_val:.1f}/MWh"
    title = f"🔮 {model_names[model_type]} - Predicción a {years_ahead} Años{gdp_label}"
    
    return pred_fig, fi_fig, season_fig, weekly_fig, message, title, metrics_div


def create_empty_figure(message: str) -> go.Figure:
    """Crea una figura vacía con mensaje."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS['text_muted'])
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig


if __name__ == '__main__':
    print("🚀 Dashboard disponible en: http://127.0.0.1:8051")
    app.run(debug=True, port=8051)
