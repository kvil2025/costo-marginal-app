"""
CMARG Dashboard Pro - Análisis Predictivo de Costo Marginal
Sistema Eléctrico Nacional de Chile
10 Barras Troncales con Predicción IA (Prophet, ARIMA, XGBoost)
"""
import os
import json
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
from flask import request, redirect, make_response, render_template, jsonify

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

# --- Configuración Auth ---
AUTH_ENABLED = os.environ.get('AUTH_ENABLED', 'true').lower() == 'true'
server = app.server
server.template_folder = os.path.join(os.path.dirname(__file__), 'templates')

if AUTH_ENABLED:
    try:
        from src.auth import (
            init_firebase, verify_firebase_token, create_or_update_user,
            check_user_authorized, get_current_user, is_admin,
            set_session_cookie, clear_session_cookie, get_session_token,
            seed_admin, list_all_users, update_user_status, update_user_role,
            delete_user, COOKIE_NAME
        )
        from src.admin import (
            create_admin_layout, render_users_table, render_stats_cards
        )
        init_firebase()
        seed_admin()  # Crear admin inicial: cristian.avila@geologgia.cl
        print("🔐 Sistema de autenticación habilitado")
    except Exception as e:
        print(f"⚠️ Auth no disponible: {e}")
        AUTH_ENABLED = False

# --- Rutas de autenticación ---
if AUTH_ENABLED:
    @server.route('/login')
    def login_page():
        """Página de login con Google Sign-In."""
        # Si ya tiene sesión válida, redirigir al dashboard
        token = request.cookies.get(COOKIE_NAME)
        if token:
            user_info = verify_firebase_token(token)
            if user_info:
                authorized, _, _ = check_user_authorized(user_info['email'])
                if authorized:
                    return redirect('/')
        return render_template('login.html')

    @server.route('/auth/callback', methods=['POST'])
    def auth_callback():
        """Recibe el token de Firebase Auth y establece la sesión."""
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token requerido'}), 400
        
        # Verificar token
        user_info = verify_firebase_token(token)
        if not user_info:
            return jsonify({'success': False, 'message': 'Token inválido'}), 401
        
        # Crear o actualizar usuario en Firestore
        user_data = create_or_update_user(user_info)
        
        # Verificar autorización
        authorized, user_data, message = check_user_authorized(user_info['email'])
        
        if authorized:
            response = make_response(jsonify({
                'success': True,
                'message': 'Acceso concedido',
                'user': {
                    'email': user_info['email'],
                    'name': user_info.get('name', ''),
                    'role': user_data.get('role', 'viewer')
                }
            }))
            set_session_cookie(response, token)
            return response
        else:
            pending = user_data and user_data.get('status') == 'pending'
            return jsonify({
                'success': False,
                'pending': pending,
                'message': message
            }), 403

    @server.route('/auth/check')
    def auth_check():
        """Verifica si hay una sesión activa."""
        user_info, user_data = get_current_user()
        if user_info and user_data:
            return jsonify({
                'authenticated': True,
                'email': user_info['email'],
                'name': user_info.get('name', ''),
                'role': user_data.get('role', 'viewer')
            })
        return jsonify({'authenticated': False})

    @server.route('/auth/logout')
    def auth_logout():
        """Cierra la sesión."""
        response = make_response(redirect('/login'))
        clear_session_cookie(response)
        return response

    @server.route('/admin')
    def admin_page():
        """Redirige a la página de admin (integrada en Dash)."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return redirect('/login')
        user_info = verify_firebase_token(token)
        if not user_info or not is_admin(user_info['email']):
            return redirect('/')
        return redirect('/admin-panel')

    # --- Admin API endpoints ---
    @server.route('/api/admin/users')
    def api_admin_users():
        """API: Lista todos los usuarios."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return jsonify({'error': 'No autorizado'}), 401
        user_info = verify_firebase_token(token)
        if not user_info or not is_admin(user_info['email']):
            return jsonify({'error': 'Acceso denegado'}), 403
        users = list_all_users()
        return jsonify({'users': users})

    @server.route('/api/admin/user/status', methods=['POST'])
    def api_admin_update_status():
        """API: Actualizar status de usuario."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return jsonify({'error': 'No autorizado'}), 401
        user_info = verify_firebase_token(token)
        if not user_info or not is_admin(user_info['email']):
            return jsonify({'error': 'Acceso denegado'}), 403
        data = request.get_json()
        success = update_user_status(data['email'], data['status'])
        return jsonify({'success': success})

    @server.route('/api/admin/user/role', methods=['POST'])
    def api_admin_update_role():
        """API: Actualizar rol de usuario."""
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return jsonify({'error': 'No autorizado'}), 401
        user_info = verify_firebase_token(token)
        if not user_info or not is_admin(user_info['email']):
            return jsonify({'error': 'Acceso denegado'}), 403
        data = request.get_json()
        success = update_user_role(data['email'], data['role'])
        return jsonify({'success': success})

    # --- Middleware: verificar autenticación antes de cada request ---
    @server.before_request
    def check_auth():
        """Middleware que verifica autenticación en todas las rutas excepto login y assets."""
        # Rutas públicas (no requieren auth)
        public_paths = ['/login', '/auth/', '/assets/', '/_dash-', '/_reload-hash', '/favicon.ico']
        path = request.path
        
        if any(path.startswith(p) for p in public_paths):
            return None
        
        # Verificar sesión
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            # Para requests AJAX de Dash, devolver 401
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
               request.content_type == 'application/json' or \
               path.startswith('/_dash'):
                return None  # Permitir Dash callbacks (la app maneja su estado)
            return redirect('/login')
        
        user_info = verify_firebase_token(token)
        if not user_info:
            if path.startswith('/_dash'):
                return None
            response = make_response(redirect('/login'))
            clear_session_cookie(response)
            return response
        
        authorized, _, _ = check_user_authorized(user_info['email'])
        if not authorized:
            if path.startswith('/_dash'):
                return None
            return redirect('/login')

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
            /* Date picker dark theme */
            .DateInput_input {
                background: #1a1a2e !important;
                color: #e2e8f0 !important;
                border: 1px solid #2d2d44 !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                font-family: 'Inter', sans-serif !important;
                font-size: 0.9rem !important;
            }
            .DateRangePickerInput {
                background: transparent !important;
                border: none !important;
            }
            .DateRangePickerInput_arrow {
                color: #94a3b8 !important;
            }
            .CalendarDay__selected {
                background: #6366f1 !important;
                border-color: #6366f1 !important;
            }
            .CalendarDay__selected_span {
                background: rgba(99, 102, 241, 0.3) !important;
                border-color: rgba(99, 102, 241, 0.3) !important;
            }
            .DayPickerNavigation_button {
                border: 1px solid #2d2d44 !important;
            }
            .year-btn {
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.3);
                color: #a5b4fc;
                padding: 5px 14px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 0.8rem;
                font-weight: 500;
                transition: all 0.2s ease;
                margin-right: 6px;
                font-family: 'Inter', sans-serif;
            }
            .year-btn:hover {
                background: rgba(99, 102, 241, 0.35);
                border-color: #6366f1;
                color: white;
            }
            .range-label {
                color: #94a3b8;
                font-size: 0.8rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
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
                        ),
                        html.Div(id='prediction-range-display', style={
                            'marginTop': '6px', 'fontSize': '11px',
                            'color': COLORS['text_muted'], 'fontFamily': 'monospace'
                        })
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
    
    # Date Range Filter
    dbc.Row([
        dbc.Col([
            html.Div(style={**CARD_STYLE, 'padding': '15px 20px'}, children=[
                dbc.Row([
                    dbc.Col([
                        html.Div("📅 Rango de Fechas", className="range-label"),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            display_format='DD/MM/YYYY',
                            start_date_placeholder_text='Desde',
                            end_date_placeholder_text='Hasta',
                            clearable=True,
                            with_portal=False,
                            first_day_of_week=1,
                            style={'width': '100%'}
                        )
                    ], md=4),
                    dbc.Col([
                        html.Div("⚡ Acceso Rápido", className="range-label"),
                        html.Div(id='year-buttons-container', children=[
                            html.Button('Todo', id='yr-btn-all', className='year-btn', n_clicks=0),
                            html.Button('2025', id='yr-btn-2025', className='year-btn', n_clicks=0),
                            html.Button('2024', id='yr-btn-2024', className='year-btn', n_clicks=0),
                            html.Button('2023', id='yr-btn-2023', className='year-btn', n_clicks=0),
                            html.Button('2022', id='yr-btn-2022', className='year-btn', n_clicks=0),
                            html.Button('2021', id='yr-btn-2021', className='year-btn', n_clicks=0),
                            html.Button('2020', id='yr-btn-2020', className='year-btn', n_clicks=0),
                            html.Button('Último Año', id='yr-btn-1y', className='year-btn', n_clicks=0, 
                                       style={'background': 'rgba(34, 211, 238, 0.15)', 'borderColor': 'rgba(34, 211, 238, 0.3)', 'color': '#22d3ee'}),
                            html.Button('Últimos 3 Años', id='yr-btn-3y', className='year-btn', n_clicks=0,
                                       style={'background': 'rgba(34, 211, 238, 0.15)', 'borderColor': 'rgba(34, 211, 238, 0.3)', 'color': '#22d3ee'}),
                        ])
                    ], md=6),
                    dbc.Col([
                        html.Div("📊 Datos Filtrados", className="range-label"),
                        html.Div(id='filter-summary', children=[
                            html.Span("Sin datos", style={'color': COLORS['text_muted'], 'fontSize': '0.85rem'})
                        ])
                    ], md=2)
                ])
            ])
        ])
    ], id='date-filter-row', style={'marginBottom': '20px', 'display': 'none'}),
    
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
    
    # === ANÁLISIS INTER-BARRA ===
    html.Div(style={'marginTop': '30px'}, children=[
        dbc.Row([
            dbc.Col([
                html.Div(style={
                    'borderTop': f'2px solid {COLORS["primary"]}',
                    'paddingTop': '20px',
                    'marginBottom': '20px'
                }, children=[
                    html.H3("🔬 Análisis Inter-Barra del SEN", style={
                        'background': f'linear-gradient(135deg, {COLORS["secondary"]}, {COLORS["accent"]})',
                        '-webkit-background-clip': 'text',
                        '-webkit-text-fill-color': 'transparent',
                        'fontWeight': '700',
                        'fontSize': '1.6rem'
                    }),
                    html.P("Spread Detector • Correlación • Detección de Anomalías", 
                           style={'color': COLORS['text_muted'], 'fontSize': '0.9rem'})
                ])
            ])
        ]),
        
        # Spread Detector Controls
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.H5("📡 Spread Detector — Diferencial Inter-Barra", className="section-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Barra A (Referencia Norte):", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='spread-bar-a',
                                options=bar_options,
                                value='crucero_220' if 'crucero_220' in loaded_bars else (bar_options[0]['value'] if bar_options else None),
                                clearable=False,
                                style={'backgroundColor': COLORS['card']}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Barra B (Referencia Sur):", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='spread-bar-b',
                                options=bar_options,
                                value='charrua_500' if 'charrua_500' in loaded_bars else (bar_options[-1]['value'] if bar_options else None),
                                clearable=False,
                                style={'backgroundColor': COLORS['card']}
                            )
                        ], md=4),
                        dbc.Col([
                            html.Label("Resolución:", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='spread-resolution',
                                options=[
                                    {'label': 'Horaria', 'value': 'H'},
                                    {'label': 'Diaria', 'value': 'D'},
                                    {'label': 'Semanal', 'value': 'W'},
                                    {'label': 'Mensual', 'value': 'M'},
                                ],
                                value='D',
                                clearable=False,
                                style={'backgroundColor': COLORS['card']}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("\u00a0", style={'fontSize': '0.85rem'}),
                            html.Button("📡 Analizar Spread", id='analyze-spread-btn', className="btn-gradient",
                                       style={'width': '100%', 'background': f'linear-gradient(135deg, {COLORS["accent"]}, #e879f9)'})
                        ], md=2)
                    ])
                ])
            ])
        ]),
        
        # Spread Chart + Anomaly KPIs
        dcc.Loading(id="loading-spread", type="circle", color=COLORS['accent'], children=[
            dbc.Row([
                dbc.Col([
                    html.Div(style=CARD_STYLE, children=[
                        html.H5(id='spread-chart-title', children="📈 Spread (A − B)", className="section-title"),
                        dcc.Graph(id='spread-plot', style={'height': '400px'},
                                 config={'displayModeBar': True, 'displaylogo': False})
                    ])
                ], md=8),
                dbc.Col([
                    html.Div(style=CARD_STYLE, children=[
                        html.H5("⚠️ Detección de Anomalías", className="section-title"),
                        html.Div(id='anomaly-summary', children=[
                            html.P("Ejecuta el análisis de spread para detectar desacoples.", 
                                   style={'color': COLORS['text_muted'], 'fontSize': '0.85rem', 'paddingTop': '20px'})
                        ])
                    ])
                ], md=4)
            ]),
            
            # Correlation Heatmap
            dbc.Row([
                dbc.Col([
                    html.Div(style=CARD_STYLE, children=[
                        html.H5("🔥 Mapa de Correlación entre Barras", className="section-title"),
                        dcc.Graph(id='correlation-heatmap', style={'height': '450px'},
                                 config={'displayModeBar': False})
                    ])
                ], md=7),
                dbc.Col([
                    html.Div(style=CARD_STYLE, children=[
                        html.H5("📊 Ranking de Volatilidad", className="section-title"),
                        dcc.Graph(id='volatility-ranking', style={'height': '450px'},
                                 config={'displayModeBar': False})
                    ])
                ], md=5)
            ])
        ])
    ]),
    
    # === GENERADOR DE INFORME DIARIO ===
    html.Div(style={'marginTop': '30px'}, children=[
        dbc.Row([
            dbc.Col([
                html.Div(style={
                    'borderTop': f'2px solid {COLORS["warning"]}',
                    'paddingTop': '20px',
                    'marginBottom': '20px'
                }, children=[
                    html.H3("📋 Generador de Informe Diario — Formato CEN", style={
                        'background': f'linear-gradient(135deg, {COLORS["warning"]}, #ef4444)',
                        '-webkit-background-clip': 'text',
                        '-webkit-text-fill-color': 'transparent',
                        'fontWeight': '700',
                        'fontSize': '1.6rem'
                    }),
                    html.P("Genera un informe con la misma estructura del Coordinador Eléctrico Nacional", 
                           style={'color': COLORS['text_muted'], 'fontSize': '0.9rem'})
                ])
            ])
        ]),
        
        # Report Controls
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Fecha del Informe:", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.DatePickerSingle(
                                id='report-date',
                                date=datetime.date.today(),
                                display_format='DD/MM/YYYY',
                                style={'width': '100%'}
                            )
                        ], md=3),
                        dbc.Col([
                            html.Label("Barras a incluir:", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='report-bars',
                                options=bar_options,
                                value=[b['value'] for b in bar_options[:5]],
                                multi=True,
                                style={'backgroundColor': COLORS['card']}
                            )
                        ], md=5),
                        dbc.Col([
                            html.Label("Tipo de Informe:", style={'color': COLORS['text'], 'fontSize': '0.85rem'}),
                            dcc.Dropdown(
                                id='report-type',
                                options=[
                                    {'label': '📊 Completo (CEN)', 'value': 'full'},
                                    {'label': '📈 Resumen Ejecutivo', 'value': 'executive'},
                                    {'label': '⚠️ Solo Anomalías', 'value': 'anomalies'},
                                ],
                                value='full',
                                clearable=False,
                                style={'backgroundColor': COLORS['card']}
                            )
                        ], md=2),
                        dbc.Col([
                            html.Label("\u00a0", style={'fontSize': '0.85rem'}),
                            html.Button("📋 Generar Informe", id='generate-report-btn', className="btn-gradient",
                                       style={'width': '100%', 'background': f'linear-gradient(135deg, {COLORS["warning"]}, #ef4444)',
                                              'padding': '10px', 'fontSize': '0.95rem'})
                        ], md=2)
                    ])
                ])
            ])
        ]),
        
        # Report Output
        dcc.Loading(id="loading-report", type="circle", color=COLORS['warning'], children=[
            dbc.Row([
                dbc.Col([
                    html.Div(id='report-output', style={
                        **CARD_STYLE,
                        'minHeight': '200px',
                        'maxHeight': '800px',
                        'overflowY': 'auto',
                        'borderLeft': f'4px solid {COLORS["warning"]}'
                    }, children=[
                        html.P("Selecciona las opciones y haz click en 'Generar Informe' para crear un reporte tipo CEN.",
                               style={'color': COLORS['text_muted'], 'padding': '40px', 'textAlign': 'center'})
                    ])
                ])
            ]),
            
            # Download button (hidden initially)
            dbc.Row([
                dbc.Col([
                    html.Div(id='report-download-section', style={'display': 'none'}, children=[
                        html.Button("⬇️ Descargar Informe HTML", id='download-report-btn', className="btn-gradient",
                                   style={'margin': '10px auto', 'display': 'block',
                                          'background': f'linear-gradient(135deg, {COLORS["accent"]}, {COLORS["secondary"]})'}),
                        dcc.Download(id='report-download')
                    ])
                ], style={'textAlign': 'center'})
            ])
        ])
    ]),
    
    # Stores
    dcc.Store(id='stored-data'),
    dcc.Store(id='prediction-data'),
    dcc.Store(id='full-data-store'),  # Stores full unfiltered data for date filtering
    dcc.Store(id='report-html-store')  # Stores generated report HTML for download
])


# --- Callbacks ---
@app.callback(
    [Output('stored-data', 'data'),
     Output('full-data-store', 'data'),
     Output('status-message', 'children'),
     Output('predict-btn', 'disabled'),
     Output('kpi-years', 'children'),
     Output('kpi-records', 'children'),
     Output('kpi-avg', 'children'),
     Output('kpi-min', 'children'),
     Output('kpi-max', 'children'),
     Output('kpi-volatility', 'children'),
     Output('historical-plot', 'figure'),
     Output('yearly-box-plot', 'figure'),
     Output('date-filter-row', 'style'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    [Input('load-btn', 'n_clicks')],
    [State('bar-selector', 'value')],
    prevent_initial_call=True
)
def load_bar_data(n_clicks, selected_bar):
    empty_extra = ({'display': 'none'}, None, None, None, None)
    if not selected_bar:
        return (None, None, "⚠️ Selecciona una barra primero.", True,
                "--", "--", "--", "--", "--", "--",
                create_empty_figure("Selecciona una barra"),
                create_empty_figure("Selecciona una barra"),
                *empty_extra)
    
    if selected_bar not in loaded_bars:
        return (None, None, f"❌ Datos no disponibles para: {selected_bar}", True,
                "--", "--", "--", "--", "--", "--",
                create_empty_figure("Sin datos"),
                create_empty_figure("Sin datos"),
                *empty_extra)
    
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
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                   rangeslider=dict(visible=True, bgcolor='rgba(26,26,46,0.5)', thickness=0.05)),
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
    
    # Guardar datos para predicción (full data always)
    stored = df.to_json(date_format='iso', orient='split')
    
    zona = info.get('zona', '')
    message = f"✅ {info.get('icon', '')} {info.get('nombre', selected_bar)} ({zona}) | {years:.1f} años | {records:,} registros"
    
    # Date range info
    min_date = df['timestamp'].min().date().isoformat()
    max_date = df['timestamp'].max().date().isoformat()
    
    return (stored, stored, message, False,
            f"{years:.1f}", f"{records:,}", f"${avg_price:.1f}",
            f"${min_price:.1f}", f"${max_price:.1f}", f"{volatility:.1f}%",
            hist_fig, box_fig,
            {'marginBottom': '20px'},  # Show date filter
            min_date, max_date, min_date, max_date)


# --- Date range filtering callback ---
@app.callback(
    [Output('historical-plot', 'figure', allow_duplicate=True),
     Output('yearly-box-plot', 'figure', allow_duplicate=True),
     Output('kpi-years', 'children', allow_duplicate=True),
     Output('kpi-records', 'children', allow_duplicate=True),
     Output('kpi-avg', 'children', allow_duplicate=True),
     Output('kpi-min', 'children', allow_duplicate=True),
     Output('kpi-max', 'children', allow_duplicate=True),
     Output('kpi-volatility', 'children', allow_duplicate=True),
     Output('filter-summary', 'children'),
     Output('date-range-picker', 'start_date', allow_duplicate=True),
     Output('date-range-picker', 'end_date', allow_duplicate=True)],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('yr-btn-all', 'n_clicks'),
     Input('yr-btn-2025', 'n_clicks'),
     Input('yr-btn-2024', 'n_clicks'),
     Input('yr-btn-2023', 'n_clicks'),
     Input('yr-btn-2022', 'n_clicks'),
     Input('yr-btn-2021', 'n_clicks'),
     Input('yr-btn-2020', 'n_clicks'),
     Input('yr-btn-1y', 'n_clicks'),
     Input('yr-btn-3y', 'n_clicks')],
    [State('full-data-store', 'data'),
     State('bar-selector', 'value')],
    prevent_initial_call=True
)
def filter_by_date(start_date, end_date, btn_all, btn_2025, btn_2024, btn_2023,
                   btn_2022, btn_2021, btn_2020, btn_1y, btn_3y, full_data, selected_bar):
    
    no_update = dash.no_update
    empty = (no_update,) * 11
    
    if not full_data:
        return empty
    
    df = pd.read_json(full_data, orient='split')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    info = BARRAS_INFO.get(selected_bar, {})
    bar_color = info.get('color', COLORS['primary'])
    
    # Determine which button was clicked
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    data_max = df['timestamp'].max()
    data_min = df['timestamp'].min()
    
    # Handle year button clicks — set date range based on button
    if triggered_id and triggered_id.startswith('yr-btn'):
        if triggered_id == 'yr-btn-all':
            start_date = data_min.date().isoformat()
            end_date = data_max.date().isoformat()
        elif triggered_id == 'yr-btn-1y':
            start_date = (data_max - pd.DateOffset(years=1)).date().isoformat()
            end_date = data_max.date().isoformat()
        elif triggered_id == 'yr-btn-3y':
            start_date = (data_max - pd.DateOffset(years=3)).date().isoformat()
            end_date = data_max.date().isoformat()
        else:
            year = int(triggered_id.split('-')[-1])
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
    
    if not start_date or not end_date:
        return empty
    
    # Filter data
    mask = (df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        summary = html.Span("Sin datos en rango", style={'color': '#ef4444', 'fontSize': '0.85rem'})
        return (create_empty_figure("Sin datos en rango seleccionado"),
                create_empty_figure("Sin datos"),
                "--", "--", "--", "--", "--", "--",
                summary, start_date, end_date)
    
    cost_col = 'costo_marginal' if 'costo_marginal' in df_filtered.columns else 'costo_usd'
    
    # Recalculate KPIs for filtered data
    years_f = (df_filtered['timestamp'].max() - df_filtered['timestamp'].min()).days / 365
    records_f = len(df_filtered)
    avg_f = df_filtered[cost_col].mean()
    min_f = df_filtered[cost_col].min()
    max_f = df_filtered[cost_col].max()
    vol_f = (df_filtered[cost_col].std() / max(avg_f, 0.01)) * 100
    
    # Rebuild historical chart
    df_daily = df_filtered.set_index('timestamp').resample('D')[cost_col].mean().reset_index()
    
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x=df_daily['timestamp'], y=df_daily[cost_col],
        mode='lines', name='CMg Diario',
        line=dict(color=bar_color, width=1.5),
        fill='tozeroy', fillcolor=f'rgba({int(bar_color[1:3],16)},{int(bar_color[3:5],16)},{int(bar_color[5:7],16)},0.1)'
    ))
    
    df_daily['ma30'] = df_daily[cost_col].rolling(window=30).mean()
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
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                   rangeslider=dict(visible=True, bgcolor='rgba(26,26,46,0.5)', thickness=0.05)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh')
    )
    
    # Rebuild box plot
    df_filtered['year'] = df_filtered['timestamp'].dt.year
    box_fig = px.box(df_filtered, x='year', y=cost_col,
                     color_discrete_sequence=[bar_color])
    box_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Año'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh'),
        showlegend=False
    )
    
    # Build filter summary
    start_fmt = pd.to_datetime(start_date).strftime('%d/%m/%Y')
    end_fmt = pd.to_datetime(end_date).strftime('%d/%m/%Y')
    summary = html.Div([
        html.Div(f"{records_f:,}", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': COLORS['secondary']}),
        html.Div("registros", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
    ])
    
    return (hist_fig, box_fig,
            f"{years_f:.1f}", f"{records_f:,}", f"${avg_f:.1f}",
            f"${min_f:.1f}", f"${max_f:.1f}", f"{vol_f:.1f}%",
            summary, start_date, end_date)


@app.callback(
    Output('prediction-range-display', 'children'),
    [Input('prediction-years-slider', 'value'),
     Input('stored-data', 'data')]
)
def update_prediction_range_display(years_ahead, stored_data):
    """Muestra el rango exacto de fechas que cubrirá la predicción."""
    from datetime import date
    import datetime
    if not years_ahead:
        return ""
    # Determinar la fecha de inicio = último dato disponible
    start = date.today()
    if stored_data:
        try:
            df = pd.DataFrame(stored_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                start = df['timestamp'].max().date()
        except Exception:
            pass
    end = start + datetime.timedelta(days=int(years_ahead) * 365)
    
    # Indicador de datos exógenos proyectados
    has_macro = end.year <= 2035
    macro_info = f" | 🌍 Macro proyectada hasta 2035 ✅" if has_macro else f" | ⚠️ Sin datos macro post-2035"
    
    # Detectar si cruza período El Niño o La Niña proyectado
    enso_notes = []
    if start.year <= 2030 and end.year >= 2028:
        enso_notes.append("☀️ El Niño 2028-2030")
    if start.year <= 2034 and end.year >= 2033:
        enso_notes.append("🌧️ La Niña 2033-2034")
    enso_str = f" | {', '.join(enso_notes)}" if enso_notes else ""
    
    return f"📅 {start.strftime('%d/%m/%Y')} → {end.strftime('%d/%m/%Y')} ({years_ahead} años){macro_info}{enso_str}"


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
     State('prediction-data', 'data'),
     State('date-range-picker', 'start_date'),
     State('date-range-picker', 'end_date')],
    prevent_initial_call=True
)
def generate_prediction(n_clicks, gdp_growth, stored_data, years_ahead, model_type, existing_prediction,
                        filter_start, filter_end):
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # === FILTRAR POR RANGO DE FECHAS SI ESTÁ ACTIVO ===
    filter_label = ""
    if filter_start and filter_end:
        fs = pd.to_datetime(filter_start)
        fe = pd.to_datetime(filter_end) + pd.Timedelta(days=1)
        mask = (df['timestamp'] >= fs) & (df['timestamp'] <= fe)
        df_filtered = df[mask].copy()
        if len(df_filtered) >= 30:  # Need minimum data points for training
            df = df_filtered
            filter_label = f" | Datos: {fs.strftime('%d/%m/%Y')} → {fe.strftime('%d/%m/%Y')} ({len(df):,} registros)"
        else:
            filter_label = f" | Rango muy corto ({len(df_filtered)} pts), usando todos los datos"
    
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
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', y=1.15),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)', 
            title='Fecha',
            rangeslider=dict(visible=True, bgcolor='rgba(26,26,46,0.5)', thickness=0.05),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1A", step="year", stepmode="backward"),
                    dict(count=2, label="2A", step="year", stepmode="backward"),
                    dict(count=3, label="3A", step="year", stepmode="backward"),
                    dict(count=5, label="5A", step="year", stepmode="backward"),
                    dict(step="all", label="Todo")
                ]),
                bgcolor='rgba(26,26,46,0.8)',
                activecolor='#6366f1',
                bordercolor='#2d2d44',
                font=dict(color='#e2e8f0', size=11),
                x=0, y=1.08
            )
        ),
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
    message = f"🔮 {model_names[model_type]}: {metrics.get('prediction_start', '')} → {metrics.get('prediction_end', '')} ({years_ahead} años){gdp_label}{filter_label} | Promedio: ${avg_val:.1f}/MWh"
    title = f"🔮 {model_names[model_type]} - Predicción a {years_ahead} Años{gdp_label}{filter_label}"
    
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


# Expose Flask server for gunicorn (Cloud Run)
server = app.server


# --- Spread Detector Callback ---
@app.callback(
    [Output('spread-plot', 'figure'),
     Output('anomaly-summary', 'children'),
     Output('spread-chart-title', 'children'),
     Output('correlation-heatmap', 'figure'),
     Output('volatility-ranking', 'figure')],
    [Input('analyze-spread-btn', 'n_clicks')],
    [State('spread-bar-a', 'value'),
     State('spread-bar-b', 'value'),
     State('spread-resolution', 'value')],
    prevent_initial_call=True
)
def analyze_spread(n_clicks, bar_a, bar_b, resolution):
    if not bar_a or not bar_b or bar_a not in loaded_bars or bar_b not in loaded_bars:
        empty = create_empty_figure("Selecciona dos barras válidas")
        return empty, "Selecciona dos barras válidas", "📈 Spread (A − B)", empty, empty
    
    # --- 1. Spread Detector ---
    df_a = loaded_bars[bar_a].copy()
    df_b = loaded_bars[bar_b].copy()
    
    # Normalize cost column
    for df in [df_a, df_b]:
        if 'costo_usd' in df.columns:
            df['costo_marginal'] = pd.to_numeric(df['costo_usd'], errors='coerce')
        else:
            df['costo_marginal'] = pd.to_numeric(df['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Resample to selected resolution
    ts_a = df_a.set_index('timestamp').resample(resolution)['costo_marginal'].mean().reset_index()
    ts_b = df_b.set_index('timestamp').resample(resolution)['costo_marginal'].mean().reset_index()
    
    # Merge on timestamp
    merged = ts_a.merge(ts_b, on='timestamp', suffixes=('_a', '_b'), how='inner')
    merged['spread'] = merged['costo_marginal_a'] - merged['costo_marginal_b']
    merged['spread_abs'] = merged['spread'].abs()
    
    # Anomaly detection (> 2 standard deviations)
    spread_mean = merged['spread'].mean()
    spread_std = merged['spread'].std()
    threshold_high = spread_mean + 2 * spread_std
    threshold_low = spread_mean - 2 * spread_std
    merged['is_anomaly'] = (merged['spread'] > threshold_high) | (merged['spread'] < threshold_low)
    anomalies = merged[merged['is_anomaly']]
    
    info_a = BARRAS_INFO.get(bar_a, {})
    info_b = BARRAS_INFO.get(bar_b, {})
    color_a = info_a.get('color', COLORS['primary'])
    color_b = info_b.get('color', COLORS['secondary'])
    
    # Build spread chart
    spread_fig = go.Figure()
    
    # Spread line
    spread_fig.add_trace(go.Scatter(
        x=merged['timestamp'], y=merged['spread'],
        mode='lines', name='Spread (A−B)',
        line=dict(width=1.5),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.1)'
    ))
    
    # Moving average of spread
    merged['spread_ma'] = merged['spread'].rolling(window=min(30, len(merged)//4 + 1)).mean()
    spread_fig.add_trace(go.Scatter(
        x=merged['timestamp'], y=merged['spread_ma'],
        mode='lines', name='Tendencia Spread',
        line=dict(color=COLORS['accent'], width=2)
    ))
    
    # Threshold bands
    spread_fig.add_hline(y=threshold_high, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)",
                         annotation_text=f"+2σ ({threshold_high:.1f})")
    spread_fig.add_hline(y=threshold_low, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)",
                         annotation_text=f"-2σ ({threshold_low:.1f})")
    spread_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    
    # Anomaly markers
    if len(anomalies) > 0:
        spread_fig.add_trace(go.Scatter(
            x=anomalies['timestamp'], y=anomalies['spread'],
            mode='markers', name=f'Anomalías ({len(anomalies)})',
            marker=dict(color='#ef4444', size=8, symbol='diamond',
                       line=dict(width=1, color='white'))
        ))
    
    spread_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', y=1.15),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                   rangeslider=dict(visible=True, bgcolor='rgba(26,26,46,0.5)', thickness=0.05)),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='USD/MWh (Diferencial)')
    )
    
    # --- 2. Anomaly Summary ---
    anomaly_children = []
    
    # KPI cards for anomaly stats
    pct_anomaly = (len(anomalies) / len(merged)) * 100 if len(merged) > 0 else 0
    max_spread = merged['spread'].max()
    min_spread = merged['spread'].min()
    max_spread_date = merged.loc[merged['spread'].idxmax(), 'timestamp'] if len(merged) > 0 else None
    
    anomaly_children.append(html.Div([
        html.Div([
            html.Div(f"{len(anomalies)}", style={'fontSize': '1.8rem', 'fontWeight': '700', 'color': '#ef4444'}),
            html.Div("Anomalías Detectadas", style={'fontSize': '0.7rem', 'color': COLORS['text_muted'], 'textTransform': 'uppercase'})
        ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(239,68,68,0.1)', 'borderRadius': '8px', 'marginBottom': '10px'}),
        
        html.Div([
            html.Div(f"{pct_anomaly:.1f}%", style={'fontSize': '1.3rem', 'fontWeight': '600', 'color': COLORS['warning']}),
            html.Div("del total de datos", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
        ], style={'textAlign': 'center', 'padding': '8px', 'background': 'rgba(245,158,11,0.1)', 'borderRadius': '8px', 'marginBottom': '10px'}),
        
        html.Hr(style={'borderColor': COLORS['card_border']}),
        
        html.Div([
            html.Div("Spread Máximo", style={'fontSize': '0.7rem', 'color': COLORS['text_muted'], 'textTransform': 'uppercase'}),
            html.Div(f"${max_spread:.1f}/MWh", style={'fontSize': '1rem', 'fontWeight': '600', 'color': '#ef4444'}),
            html.Div(f"{max_spread_date.strftime('%d/%m/%Y') if max_spread_date else 'N/A'}", 
                     style={'fontSize': '0.75rem', 'color': COLORS['text_muted']})
        ], style={'marginBottom': '8px'}),
        
        html.Div([
            html.Div("Spread Mínimo", style={'fontSize': '0.7rem', 'color': COLORS['text_muted'], 'textTransform': 'uppercase'}),
            html.Div(f"${min_spread:.1f}/MWh", style={'fontSize': '1rem', 'fontWeight': '600', 'color': COLORS['secondary']}),
        ], style={'marginBottom': '8px'}),
        
        html.Div([
            html.Div("Promedio ± σ", style={'fontSize': '0.7rem', 'color': COLORS['text_muted'], 'textTransform': 'uppercase'}),
            html.Div(f"${spread_mean:.1f} ± ${spread_std:.1f}", 
                     style={'fontSize': '1rem', 'fontWeight': '600', 'color': COLORS['primary']}),
        ]),
    ]))
    
    # --- 3. Correlation Heatmap ---
    # Build daily average price matrix for all bars
    price_matrix = {}
    for bid, bdf in loaded_bars.items():
        tmp = bdf.copy()
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        if 'costo_usd' in tmp.columns:
            tmp['price'] = pd.to_numeric(tmp['costo_usd'], errors='coerce')
        else:
            tmp['price'] = pd.to_numeric(tmp['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
        daily = tmp.set_index('timestamp').resample('D')['price'].mean()
        binfo = BARRAS_INFO.get(bid, {})
        short_name = binfo.get('zona', bid.replace('_', ' ').title())
        price_matrix[short_name] = daily
    
    price_df = pd.DataFrame(price_matrix).dropna()
    
    if len(price_df.columns) >= 2:
        corr = price_df.corr()
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, '#ef4444'], [0.5, '#1a1a2e'], [1, '#6366f1']],
            zmid=0.85,
            text=[[f'{v:.2f}' for v in row] for row in corr.values],
            texttemplate='%{text}',
            textfont=dict(size=10, color='white'),
            hovertemplate='%{x} vs %{y}<br>Correlación: %{z:.3f}<extra></extra>'
        ))
        heatmap_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9))
        )
    else:
        heatmap_fig = create_empty_figure("Necesita al menos 2 barras cargadas")
    
    # --- 4. Volatility Ranking ---
    vol_data = []
    for bid, bdf in loaded_bars.items():
        tmp = bdf.copy()
        if 'costo_usd' in tmp.columns:
            prices = pd.to_numeric(tmp['costo_usd'], errors='coerce')
        else:
            prices = pd.to_numeric(tmp['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
        avg = prices.mean()
        vol = (prices.std() / max(avg, 0.01)) * 100
        binfo = BARRAS_INFO.get(bid, {})
        vol_data.append({
            'barra': binfo.get('zona', bid),
            'volatilidad': vol,
            'color': binfo.get('color', COLORS['primary'])
        })
    
    vol_df = pd.DataFrame(vol_data).sort_values('volatilidad', ascending=True)
    
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(
        x=vol_df['volatilidad'],
        y=vol_df['barra'],
        orientation='h',
        marker_color=vol_df['color'].tolist(),
        text=[f'{v:.1f}%' for v in vol_df['volatilidad']],
        textposition='outside',
        textfont=dict(color=COLORS['text'], size=11)
    ))
    vol_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=50, t=10, b=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Volatilidad (CV %)', range=[0, vol_df['volatilidad'].max() * 1.3]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    # Title
    name_a = info_a.get('zona', bar_a)
    name_b = info_b.get('zona', bar_b)
    title = f"📈 Spread: {name_a} → {name_b} | {len(merged):,} puntos | {len(anomalies)} anomalías"
    
    return spread_fig, anomaly_children, title, heatmap_fig, vol_fig


# --- Report Generator Callback ---
@app.callback(
    [Output('report-output', 'children'),
     Output('report-download-section', 'style'),
     Output('report-html-store', 'data')],
    [Input('generate-report-btn', 'n_clicks')],
    [State('report-date', 'date'),
     State('report-bars', 'value'),
     State('report-type', 'value')],
    prevent_initial_call=True
)
def generate_report(n_clicks, report_date, selected_bars, report_type):
    if not selected_bars:
        return [html.P("⚠️ Selecciona al menos una barra.", style={'color': COLORS['warning'], 'padding': '20px'})], {'display': 'none'}, None
    
    report_date = pd.to_datetime(report_date)
    date_str = report_date.strftime('%d de %B de %Y').replace('January', 'enero').replace('February', 'febrero').replace('March', 'marzo').replace('April', 'abril').replace('May', 'mayo').replace('June', 'junio').replace('July', 'julio').replace('August', 'agosto').replace('September', 'septiembre').replace('October', 'octubre').replace('November', 'noviembre').replace('December', 'diciembre')
    day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    day_name = day_names.get(report_date.weekday(), '')
    
    header_style = {'background': f'linear-gradient(135deg, {COLORS["warning"]}, #ef4444)',
                    'padding': '15px 20px', 'borderRadius': '10px', 'marginBottom': '20px'}
    section_style = {'background': 'rgba(26,26,46,0.6)', 'padding': '15px', 'borderRadius': '8px',
                     'marginBottom': '15px', 'border': f'1px solid {COLORS["card_border"]}'}
    table_header = {'backgroundColor': 'rgba(99,102,241,0.2)', 'color': COLORS['text'],
                    'padding': '8px 12px', 'textAlign': 'left', 'fontSize': '0.8rem', 'fontWeight': '600'}
    table_cell = {'padding': '6px 12px', 'borderBottom': f'1px solid {COLORS["card_border"]}',
                  'color': COLORS['text'], 'fontSize': '0.8rem'}
    
    report_children = []
    html_download = []
    
    # === HEADER ===
    report_children.append(html.Div(style=header_style, children=[
        html.H2(f"INFORME DIARIO", style={'color': 'white', 'margin': '0', 'fontWeight': '800'}),
        html.H4(f"{day_name} {date_str}", style={'color': 'rgba(255,255,255,0.9)', 'margin': '5px 0 0 0', 'fontWeight': '400'}),
        html.P("Coordinador Eléctrico Nacional — Generado por CMARG Pro", 
               style={'color': 'rgba(255,255,255,0.6)', 'margin': '5px 0 0 0', 'fontSize': '0.8rem'})
    ]))
    
    # === 1. RESUMEN DE COSTOS MARGINALES ===
    report_children.append(html.Div(style=section_style, children=[
        html.H5("1. Resumen de Costos Marginales", style={'color': COLORS['primary'], 'fontWeight': '700',
                                                            'borderBottom': f'2px solid {COLORS["primary"]}', 'paddingBottom': '8px'})
    ]))
    
    bar_stats = []
    for bid in selected_bars:
        if bid not in loaded_bars:
            continue
        df = loaded_bars[bid].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'costo_usd' in df.columns:
            df['price'] = pd.to_numeric(df['costo_usd'], errors='coerce')
        else:
            df['price'] = pd.to_numeric(df['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Filter to report date range: last 7 days
        date_mask = (df['timestamp'].dt.date >= (report_date - pd.Timedelta(days=7)).date()) & \
                    (df['timestamp'].dt.date <= report_date.date())
        df_period = df[date_mask]
        
        # Day-specific stats
        day_mask = df['timestamp'].dt.date == report_date.date()
        df_day = df[day_mask]
        
        info = BARRAS_INFO.get(bid, {})
        
        if len(df_day) > 0:
            avg_day = df_day['price'].mean()
            max_day = df_day['price'].max()
            min_day = df_day['price'].min()
            vol = (df_day['price'].std() / max(avg_day, 0.01)) * 100
        elif len(df_period) > 0:
            avg_day = df_period['price'].mean()
            max_day = df_period['price'].max()
            min_day = df_period['price'].min()
            vol = (df_period['price'].std() / max(avg_day, 0.01)) * 100
        else:
            avg_day = df['price'].mean()
            max_day = df['price'].max()
            min_day = df['price'].min()
            vol = (df['price'].std() / max(avg_day, 0.01)) * 100
        
        bar_stats.append({
            'barra': info.get('nombre', bid),
            'zona': info.get('zona', ''),
            'promedio': avg_day,
            'maximo': max_day,
            'minimo': min_day,
            'volatilidad': vol,
            'registros': len(df_day) if len(df_day) > 0 else len(df_period),
            'color': info.get('color', COLORS['primary'])
        })
    
    # Cost summary table
    if bar_stats:
        table_rows = [html.Tr([
            html.Th("Barra", style=table_header),
            html.Th("Zona", style=table_header),
            html.Th("Promedio (USD/MWh)", style=table_header),
            html.Th("Máximo", style=table_header),
            html.Th("Mínimo", style=table_header),
            html.Th("Volatilidad", style=table_header),
        ])]
        for s in bar_stats:
            vol_color = '#ef4444' if s['volatilidad'] > 80 else (COLORS['warning'] if s['volatilidad'] > 50 else COLORS['success'])
            table_rows.append(html.Tr([
                html.Td(s['barra'], style={**table_cell, 'color': s['color'], 'fontWeight': '600'}),
                html.Td(s['zona'], style=table_cell),
                html.Td(f"${s['promedio']:.2f}", style={**table_cell, 'fontWeight': '600'}),
                html.Td(f"${s['maximo']:.2f}", style={**table_cell, 'color': '#ef4444'}),
                html.Td(f"${s['minimo']:.2f}", style={**table_cell, 'color': COLORS['success']}),
                html.Td(f"{s['volatilidad']:.1f}%", style={**table_cell, 'color': vol_color}),
            ]))
        report_children.append(html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}))
    
    if report_type in ['full', 'anomalies']:
        # === 2. DETECCIÓN DE ANOMALÍAS Y SPREADS ===
        report_children.append(html.Div(style=section_style, children=[
            html.H5("2. Detección de Anomalías y Spreads Inter-Barra", style={'color': '#ef4444', 'fontWeight': '700',
                                                                                'borderBottom': '2px solid #ef4444', 'paddingBottom': '8px'})
        ]))
        
        anomaly_report = []
        pairs_checked = 0
        total_anomalies = 0
        
        bar_ids = [b for b in selected_bars if b in loaded_bars]
        for i in range(len(bar_ids)):
            for j in range(i + 1, len(bar_ids)):
                ba, bb = bar_ids[i], bar_ids[j]
                df_a = loaded_bars[ba].copy()
                df_b = loaded_bars[bb].copy()
                
                for df in [df_a, df_b]:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if 'costo_usd' in df.columns:
                        df['price'] = pd.to_numeric(df['costo_usd'], errors='coerce')
                    else:
                        df['price'] = pd.to_numeric(df['costo_marginal'].astype(str).str.replace(',', '.'), errors='coerce')
                
                ts_a = df_a.set_index('timestamp').resample('D')['price'].mean().reset_index()
                ts_b = df_b.set_index('timestamp').resample('D')['price'].mean().reset_index()
                merged = ts_a.merge(ts_b, on='timestamp', suffixes=('_a', '_b'), how='inner')
                merged['spread'] = merged['price_a'] - merged['price_b']
                
                if len(merged) < 10:
                    continue
                    
                spread_mean = merged['spread'].mean()
                spread_std = merged['spread'].std()
                threshold = spread_mean + 2 * spread_std
                threshold_low = spread_mean - 2 * spread_std
                n_anom = int(((merged['spread'] > threshold) | (merged['spread'] < threshold_low)).sum())
                
                pairs_checked += 1
                total_anomalies += n_anom
                
                if n_anom > 0:
                    max_spread = merged['spread'].max()
                    max_date = merged.loc[merged['spread'].idxmax(), 'timestamp']
                    info_a = BARRAS_INFO.get(ba, {})
                    info_b = BARRAS_INFO.get(bb, {})
                    anomaly_report.append({
                        'par': f"{info_a.get('zona', ba)} ↔ {info_b.get('zona', bb)}",
                        'anomalias': n_anom,
                        'spread_max': max_spread,
                        'fecha_max': max_date.strftime('%d/%m/%Y'),
                        'promedio': spread_mean,
                        'umbral': threshold
                    })
        
        # Summary KPIs
        report_children.append(dbc.Row([
            dbc.Col(html.Div([
                html.Div(f"{pairs_checked}", style={'fontSize': '1.5rem', 'fontWeight': '700', 'color': COLORS['primary']}),
                html.Div("Pares Analizados", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
            ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(99,102,241,0.1)', 'borderRadius': '8px'}), md=4),
            dbc.Col(html.Div([
                html.Div(f"{total_anomalies}", style={'fontSize': '1.5rem', 'fontWeight': '700', 'color': '#ef4444'}),
                html.Div("Anomalías Totales", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
            ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(239,68,68,0.1)', 'borderRadius': '8px'}), md=4),
            dbc.Col(html.Div([
                html.Div(f"{len(anomaly_report)}", style={'fontSize': '1.5rem', 'fontWeight': '700', 'color': COLORS['warning']}),
                html.Div("Pares con Desacople", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
            ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(245,158,11,0.1)', 'borderRadius': '8px'}), md=4)
        ], style={'marginBottom': '15px'}))
        
        if anomaly_report:
            anom_sorted = sorted(anomaly_report, key=lambda x: x['anomalias'], reverse=True)
            anom_rows = [html.Tr([
                html.Th("Par de Barras", style=table_header),
                html.Th("Anomalías", style=table_header),
                html.Th("Spread Máx", style=table_header),
                html.Th("Fecha", style=table_header),
                html.Th("Promedio", style=table_header),
                html.Th("Umbral 2σ", style=table_header),
            ])]
            for a in anom_sorted[:10]:
                anom_rows.append(html.Tr([
                    html.Td(a['par'], style={**table_cell, 'fontWeight': '600'}),
                    html.Td(str(a['anomalias']), style={**table_cell, 'color': '#ef4444', 'fontWeight': '700'}),
                    html.Td(f"${a['spread_max']:.1f}", style={**table_cell, 'color': '#ef4444'}),
                    html.Td(a['fecha_max'], style=table_cell),
                    html.Td(f"${a['promedio']:.1f}", style=table_cell),
                    html.Td(f"${a['umbral']:.1f}", style={**table_cell, 'color': COLORS['warning']}),
                ]))
            report_children.append(html.Table(anom_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}))
    
    if report_type == 'full':
        # === 3. ESTADO DE GENERACIÓN ===
        cen_dir = Path('data/cen_reports/2026-03-17')
        if cen_dir.exists():
            report_children.append(html.Div(style=section_style, children=[
                html.H5("3. Estado de Generación — Desviaciones Real vs Programada", 
                         style={'color': COLORS['success'], 'fontWeight': '700',
                                'borderBottom': f'2px solid {COLORS["success"]}', 'paddingBottom': '8px'}),
                html.P("Fuente: Informe CEN 17 marzo 2026", style={'color': COLORS['text_muted'], 'fontSize': '0.75rem'})
            ]))
            
            # Load generation data from CEN tables
            gen_data = []
            for table_num in [6, 7, 8, 9, 10]:
                table_path = cen_dir / f'table_{table_num}.csv'
                if table_path.exists():
                    try:
                        tdf = pd.read_csv(table_path, header=None)
                        for _, row in tdf.iterrows():
                            vals = row.values
                            if len(vals) >= 4:
                                name = str(vals[0]).strip() if pd.notna(vals[0]) else ''
                                if name and not name.startswith('1.') and name != '0' and 'Central' not in name:
                                    try:
                                        prog = float(str(vals[1]).replace(',', ''))
                                        real = float(str(vals[2]).replace(',', ''))
                                        desv_str = str(vals[3]).replace('%', '').replace(' ', '') if pd.notna(vals[3]) else ''
                                        if desv_str:
                                            desv = float(desv_str)
                                        else:
                                            desv = ((real - prog) / max(prog, 0.01)) * 100 if prog > 0 else 0
                                        
                                        if prog > 0 or real > 0:
                                            gen_data.append({
                                                'central': name, 'prog': prog, 'real': real, 'desv': desv
                                            })
                                    except (ValueError, TypeError):
                                        pass
                            # Also check columns 5-9 for the second set
                            if len(vals) >= 9:
                                name2 = str(vals[5]).strip() if pd.notna(vals[5]) else ''
                                if name2 and 'Central' not in name2 and 'TOTAL' not in name2:
                                    try:
                                        prog2 = float(str(vals[6]).replace(',', ''))
                                        real2 = float(str(vals[7]).replace(',', ''))
                                        desv_str2 = str(vals[8]).replace('%', '').replace(' ', '') if pd.notna(vals[8]) else ''
                                        if desv_str2:
                                            desv2 = float(desv_str2)
                                        else:
                                            desv2 = ((real2 - prog2) / max(prog2, 0.01)) * 100 if prog2 > 0 else 0
                                        
                                        if prog2 > 0 or real2 > 0:
                                            gen_data.append({
                                                'central': name2, 'prog': prog2, 'real': real2, 'desv': desv2
                                            })
                                    except (ValueError, TypeError):
                                        pass
                    except Exception:
                        pass
            
            if gen_data:
                gen_df = pd.DataFrame(gen_data)
                total_prog = gen_df['prog'].sum()
                total_real = gen_df['real'].sum()
                total_desv = ((total_real - total_prog) / max(total_prog, 0.01)) * 100
                
                report_children.append(dbc.Row([
                    dbc.Col(html.Div([
                        html.Div(f"{total_prog:,.0f} MWh", style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': COLORS['primary']}),
                        html.Div("Programado", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
                    ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(99,102,241,0.1)', 'borderRadius': '8px'}), md=3),
                    dbc.Col(html.Div([
                        html.Div(f"{total_real:,.0f} MWh", style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': COLORS['success']}),
                        html.Div("Real", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
                    ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(16,185,129,0.1)', 'borderRadius': '8px'}), md=3),
                    dbc.Col(html.Div([
                        html.Div(f"{total_desv:+.2f}%", style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': '#ef4444' if total_desv < 0 else COLORS['success']}),
                        html.Div("Desviación", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
                    ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(239,68,68,0.1)', 'borderRadius': '8px'}), md=3),
                    dbc.Col(html.Div([
                        html.Div(f"{len(gen_data)}", style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': COLORS['warning']}),
                        html.Div("Centrales Activas", style={'fontSize': '0.7rem', 'color': COLORS['text_muted']})
                    ], style={'textAlign': 'center', 'padding': '10px', 'background': 'rgba(245,158,11,0.1)', 'borderRadius': '8px'}), md=3)
                ], style={'marginBottom': '15px'}))
                
                # Top deviations (positive & negative)
                gen_df_sorted = gen_df.sort_values('desv')
                worst_neg = gen_df_sorted.head(10)
                worst_pos = gen_df_sorted.tail(5).iloc[::-1]
                
                # Negative deviations table
                report_children.append(html.H6("🔻 Mayores déficits de generación:", style={'color': '#ef4444', 'marginTop': '10px'}))
                neg_rows = [html.Tr([
                    html.Th("Central", style=table_header), html.Th("Prog. (MWh)", style=table_header),
                    html.Th("Real (MWh)", style=table_header), html.Th("Desviación", style=table_header)
                ])]
                for _, r in worst_neg.iterrows():
                    neg_rows.append(html.Tr([
                        html.Td(r['central'], style={**table_cell, 'fontWeight': '600'}),
                        html.Td(f"{r['prog']:,.1f}", style=table_cell),
                        html.Td(f"{r['real']:,.1f}", style=table_cell),
                        html.Td(f"{r['desv']:+.1f}%", style={**table_cell, 'color': '#ef4444', 'fontWeight': '700'})
                    ]))
                report_children.append(html.Table(neg_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}))
                
                # Positive deviations
                report_children.append(html.H6("🔺 Mayor sobreproducción:", style={'color': COLORS['success'], 'marginTop': '10px'}))
                pos_rows = [html.Tr([
                    html.Th("Central", style=table_header), html.Th("Prog. (MWh)", style=table_header),
                    html.Th("Real (MWh)", style=table_header), html.Th("Desviación", style=table_header)
                ])]
                for _, r in worst_pos.iterrows():
                    pos_rows.append(html.Tr([
                        html.Td(r['central'], style={**table_cell, 'fontWeight': '600'}),
                        html.Td(f"{r['prog']:,.1f}", style=table_cell),
                        html.Td(f"{r['real']:,.1f}", style=table_cell),
                        html.Td(f"{r['desv']:+.1f}%", style={**table_cell, 'color': COLORS['success'], 'fontWeight': '700'})
                    ]))
                report_children.append(html.Table(pos_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}))
        
        # === 4. MANTENIMIENTO MAYOR ===
        maint_path = Path('data/cen_reports/2026-03-17/table_20.csv')
        if maint_path.exists():
            report_children.append(html.Div(style=section_style, children=[
                html.H5("4. Mantenimiento Mayor — Centrales ≥100 MW",
                         style={'color': COLORS['warning'], 'fontWeight': '700',
                                'borderBottom': f'2px solid {COLORS["warning"]}', 'paddingBottom': '8px'})
            ]))
            try:
                mdf = pd.read_csv(maint_path, header=None)
                maint_rows = [html.Tr([
                    html.Th("Central", style=table_header),
                    html.Th("Disponibilidad", style=table_header),
                    html.Th("Observaciones", style=table_header),
                ])]
                for _, row in mdf.iterrows():
                    vals = row.values
                    name = str(vals[0]).strip() if pd.notna(vals[0]) else ''
                    if name and 'CENTRAL' not in name and 'Mantenimiento' not in name:
                        dispo = str(vals[1]).strip() if len(vals) > 1 and pd.notna(vals[1]) else ''
                        obs = str(vals[2]).strip() if len(vals) > 2 and pd.notna(vals[2]) else ''
                        dispo_color = '#ef4444' if '50' in dispo else (COLORS['warning'] if '89' in dispo or '57' in dispo else COLORS['success'])
                        maint_rows.append(html.Tr([
                            html.Td(name, style={**table_cell, 'fontWeight': '600'}),
                            html.Td(dispo, style={**table_cell, 'color': dispo_color, 'fontWeight': '700'}),
                            html.Td(obs, style={**table_cell, 'fontSize': '0.7rem'}),
                        ]))
                report_children.append(html.Table(maint_rows, style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}))
            except Exception:
                pass
    
    # === FOOTER ===
    report_children.append(html.Div(style={
        'borderTop': f'1px solid {COLORS["card_border"]}',
        'paddingTop': '15px', 'marginTop': '20px'
    }, children=[
        html.P(f"📋 Informe generado el {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')} por CMARG Pro v2.0",
               style={'color': COLORS['text_muted'], 'fontSize': '0.75rem', 'textAlign': 'center'}),
        html.P("Sistema Eléctrico Nacional — Análisis basado en datos del Coordinador Eléctrico Nacional",
               style={'color': COLORS['text_muted'], 'fontSize': '0.7rem', 'textAlign': 'center'})
    ]))
    
    return report_children, {'display': 'block'}, f"report_{report_date.strftime('%Y%m%d')}.html"


# --- Download Report Callback ---
@app.callback(
    Output('report-download', 'data'),
    [Input('download-report-btn', 'n_clicks')],
    [State('report-html-store', 'data'),
     State('report-date', 'date')],
    prevent_initial_call=True
)
def download_report(n_clicks, stored_filename, report_date):
    if not stored_filename:
        return None
    report_date = pd.to_datetime(report_date)
    content = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>Informe Diario CMARG Pro - {report_date.strftime('%d/%m/%Y')}</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#0f0f23;color:#e2e8f0;padding:40px;}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px;}}
th{{background:rgba(99,102,241,0.3);padding:10px;text-align:left;font-size:0.85rem;}}
td{{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.1);font-size:0.85rem;}}
h2{{color:#f59e0b;}} h5{{color:#6366f1;border-bottom:2px solid #6366f1;padding-bottom:8px;}}
.kpi{{text-align:center;padding:15px;border-radius:8px;display:inline-block;margin:5px;min-width:150px;}}
</style></head><body>
<h2>INFORME DIARIO — {report_date.strftime('%d/%m/%Y')}</h2>
<p>Generado por CMARG Pro | Datos del Coordinador Eléctrico Nacional</p>
<p style='color:#888;font-size:0.8rem;'>Para ver el informe interactivo, visite el dashboard en Cloud Run.</p>
</body></html>"""
    return dict(content=content, filename=stored_filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8051))
    debug = os.environ.get('DASH_DEBUG', 'true').lower() == 'true'
    auth_status = '🔐 Auth HABILITADO' if AUTH_ENABLED else '🔓 Auth DESHABILITADO'
    print(f"{auth_status}")
    print(f"🚀 Dashboard disponible en: http://127.0.0.1:{port}")
    if AUTH_ENABLED:
        print(f"🔑 Login en: http://127.0.0.1:{port}/login")
    app.run(debug=debug, port=port, host='0.0.0.0')

