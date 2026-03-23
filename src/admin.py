"""
CMARG Pro — Admin Panel
Panel de administración para gestionar usuarios.
Solo accesible para usuarios con role 'admin'.
"""
import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
from datetime import datetime


def create_admin_layout():
    """Crea el layout del panel de administración."""
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div("🛡️", style={'fontSize': '32px'}),
                html.Div([
                    html.H2("Panel de Administración", style={
                        'margin': '0',
                        'background': 'linear-gradient(135deg, #60a5fa, #34d399)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'fontSize': '24px'
                    }),
                    html.P("Gestión de usuarios CMARG Pro", style={
                        'margin': '0',
                        'color': 'rgba(255,255,255,0.4)',
                        'fontSize': '13px'
                    })
                ]),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'gap': '16px'
            }),
            html.Div([
                dbc.Button("🔄 Actualizar", id='admin-refresh-btn', 
                          color='primary', outline=True, size='sm',
                          style={'marginRight': '8px'}),
                dbc.Button("← Volver al Dashboard", id='admin-back-btn',
                          color='secondary', outline=True, size='sm',
                          href='/')
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'padding': '20px 30px',
            'background': 'rgba(15, 20, 35, 0.6)',
            'borderBottom': '1px solid rgba(255,255,255,0.06)',
            'borderRadius': '16px 16px 0 0'
        }),

        # Stats cards
        html.Div(id='admin-stats-cards', style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(4, 1fr)',
            'gap': '16px',
            'padding': '20px 30px'
        }),

        # Users table
        html.Div([
            html.H4("👥 Usuarios Registrados", style={
                'color': '#fff',
                'marginBottom': '16px',
                'fontSize': '18px'
            }),
            html.Div(id='admin-users-table', style={
                'overflowX': 'auto'
            })
        ], style={
            'padding': '20px 30px'
        }),

        # Hidden stores
        dcc.Store(id='admin-users-data'),
        dcc.Interval(id='admin-refresh-interval', interval=30000, n_intervals=0),  # Auto-refresh 30s
        
        # Confirm dialog
        dcc.ConfirmDialog(id='admin-confirm-dialog', message=''),

    ], style={
        'background': 'rgba(10, 14, 26, 0.95)',
        'borderRadius': '16px',
        'border': '1px solid rgba(255,255,255,0.06)',
        'margin': '20px',
        'minHeight': '80vh'
    })


def _stat_card(title, value, icon, color):
    """Crea una tarjeta de estadísticas."""
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '24px'}),
            html.Div([
                html.Div(str(value), style={
                    'fontSize': '28px',
                    'fontWeight': '700',
                    'color': color
                }),
                html.Div(title, style={
                    'fontSize': '12px',
                    'color': 'rgba(255,255,255,0.4)',
                    'textTransform': 'uppercase',
                    'letterSpacing': '1px'
                })
            ])
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'gap': '14px'
        })
    ], style={
        'background': 'rgba(255,255,255,0.03)',
        'border': '1px solid rgba(255,255,255,0.06)',
        'borderRadius': '14px',
        'padding': '20px',
    })


def _user_row(user, idx):
    """Crea una fila de usuario para la tabla."""
    email = user.get('email', 'N/A')
    name = user.get('name', 'Sin nombre')
    role = user.get('role', 'viewer')
    status = user.get('status', 'pending')
    last_login = user.get('last_login', 'Nunca')
    created = user.get('created_at', 'N/A')
    picture = user.get('picture', '')

    # Status badge
    status_colors = {
        'active': ('#10b981', '✅'),
        'pending': ('#f59e0b', '⏳'),
        'blocked': ('#ef4444', '🚫')
    }
    s_color, s_icon = status_colors.get(status, ('#6b7280', '❓'))

    # Role badge
    role_colors = {
        'admin': '#8b5cf6',
        'analyst': '#3b82f6',
        'viewer': '#6b7280'
    }
    r_color = role_colors.get(role, '#6b7280')

    # Format dates
    if last_login and last_login != 'Nunca' and last_login != '':
        try:
            dt = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
            last_login = dt.strftime('%d/%m/%Y %H:%M')
        except:
            pass
    
    if not last_login or last_login == '':
        last_login = 'Nunca'

    return html.Tr([
        # Avatar + Name
        html.Td(
            html.Div([
                html.Img(src=picture, style={
                    'width': '32px', 'height': '32px',
                    'borderRadius': '50%', 'objectFit': 'cover'
                }) if picture else html.Div("👤", style={'fontSize': '24px'}),
                html.Div([
                    html.Div(name, style={'color': '#fff', 'fontWeight': '500', 'fontSize': '14px'}),
                    html.Div(email, style={'color': 'rgba(255,255,255,0.4)', 'fontSize': '12px'})
                ])
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
            style={'padding': '12px 16px'}
        ),
        # Role
        html.Td(
            html.Span(role.upper(), style={
                'background': f'{r_color}22',
                'color': r_color,
                'padding': '4px 12px',
                'borderRadius': '20px',
                'fontSize': '11px',
                'fontWeight': '600',
                'letterSpacing': '0.5px'
            }),
            style={'padding': '12px 16px'}
        ),
        # Status
        html.Td(
            html.Span(f"{s_icon} {status}", style={
                'color': s_color,
                'fontSize': '13px',
                'fontWeight': '500'
            }),
            style={'padding': '12px 16px'}
        ),
        # Last login
        html.Td(last_login, style={
            'padding': '12px 16px',
            'color': 'rgba(255,255,255,0.5)',
            'fontSize': '13px'
        }),
        # Actions
        html.Td(
            html.Div([
                # Status toggle
                dbc.Button(
                    "✅ Activar" if status != 'active' else "🚫 Bloquear",
                    id={'type': 'admin-status-btn', 'index': email},
                    color='success' if status != 'active' else 'danger',
                    size='sm',
                    outline=True,
                    style={'marginRight': '4px', 'fontSize': '11px'}
                ),
                # Role dropdown
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("👑 Admin", id={'type': 'admin-role-admin', 'index': email}),
                    dbc.DropdownMenuItem("📊 Analyst", id={'type': 'admin-role-analyst', 'index': email}),
                    dbc.DropdownMenuItem("👁️ Viewer", id={'type': 'admin-role-viewer', 'index': email}),
                ], label="Rol", size='sm', color='info',
                   toggle_style={'fontSize': '11px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}),
            style={'padding': '12px 16px'}
        )
    ], style={
        'borderBottom': '1px solid rgba(255,255,255,0.04)',
        'transition': 'background 0.2s',
    })


def render_users_table(users):
    """Renderiza la tabla de usuarios."""
    if not users:
        return html.Div("No hay usuarios registrados.", style={
            'color': 'rgba(255,255,255,0.4)',
            'padding': '40px',
            'textAlign': 'center'
        })

    header = html.Thead(html.Tr([
        html.Th("Usuario", style={'padding': '12px 16px', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Th("Rol", style={'padding': '12px 16px', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Th("Estado", style={'padding': '12px 16px', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Th("Último Login", style={'padding': '12px 16px', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Th("Acciones", style={'padding': '12px 16px', 'color': 'rgba(255,255,255,0.5)', 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
    ], style={'borderBottom': '1px solid rgba(255,255,255,0.08)'}))

    body = html.Tbody([_user_row(u, i) for i, u in enumerate(users)])

    return html.Table([header, body], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'background': 'rgba(255,255,255,0.02)',
        'borderRadius': '12px',
        'overflow': 'hidden'
    })


def render_stats_cards(users):
    """Renderiza las tarjetas de estadísticas."""
    total = len(users)
    active = sum(1 for u in users if u.get('status') == 'active')
    pending = sum(1 for u in users if u.get('status') == 'pending')
    admins = sum(1 for u in users if u.get('role') == 'admin')

    return [
        _stat_card("Total Usuarios", total, "👥", "#60a5fa"),
        _stat_card("Activos", active, "✅", "#10b981"),
        _stat_card("Pendientes", pending, "⏳", "#f59e0b"),
        _stat_card("Administradores", admins, "👑", "#8b5cf6"),
    ]
