"""
CMARG Pro — Firebase Authentication Module
Verifica tokens de Google Sign-In via Firebase Admin SDK
y gestiona autorización de usuarios mediante Firestore.
"""
import os
import json
import functools
from datetime import datetime
from flask import request, redirect, make_response, render_template, jsonify

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore

# ══════════════════════════════════════════════════════════════
# Inicialización Firebase Admin
# ══════════════════════════════════════════════════════════════

def init_firebase():
    """Inicializa Firebase Admin SDK si no está ya inicializado."""
    if not firebase_admin._apps:
        # En Cloud Run usa Application Default Credentials
        # En local puede usar GOOGLE_APPLICATION_CREDENTIALS
        try:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'projectId': os.environ.get('GCP_PROJECT_ID', 'geologgia-map')
            })
            print("✅ Firebase Admin SDK inicializado")
        except Exception as e:
            print(f"⚠️ Firebase Admin SDK error: {e}")
            # Fallback: inicializar sin credenciales explícitas
            try:
                firebase_admin.initialize_app(options={
                    'projectId': os.environ.get('GCP_PROJECT_ID', 'geologgia-map')
                })
                print("✅ Firebase Admin SDK inicializado (fallback)")
            except Exception as e2:
                print(f"❌ No se pudo inicializar Firebase: {e2}")

# Firestore client (lazy initialization)
_db = None

def get_db():
    """Obtiene el cliente Firestore."""
    global _db
    if _db is None:
        init_firebase()
        _db = firestore.client()
    return _db


# ══════════════════════════════════════════════════════════════
# Verificación de Tokens
# ══════════════════════════════════════════════════════════════

def verify_firebase_token(id_token: str) -> dict:
    """
    Verifica un ID token de Firebase Auth.
    Returns: dict con info del usuario o None si inválido.
    """
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return {
            'uid': decoded['uid'],
            'email': decoded.get('email', ''),
            'name': decoded.get('name', ''),
            'picture': decoded.get('picture', ''),
            'email_verified': decoded.get('email_verified', False)
        }
    except firebase_auth.InvalidIdTokenError:
        print("❌ Token inválido")
        return None
    except firebase_auth.ExpiredIdTokenError:
        print("❌ Token expirado")
        return None
    except Exception as e:
        print(f"❌ Error verificando token: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# Gestión de Usuarios en Firestore
# ══════════════════════════════════════════════════════════════

USERS_COLLECTION = 'cmarg_users'

def get_user(email: str) -> dict:
    """Obtiene un usuario de Firestore por email."""
    try:
        db = get_db()
        doc = db.collection(USERS_COLLECTION).document(email).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        print(f"❌ Error obteniendo usuario: {e}")
        return None

def create_or_update_user(user_info: dict) -> dict:
    """
    Crea o actualiza un usuario en Firestore.
    Si es nuevo, lo crea con status 'pending'.
    Si existe, actualiza last_login.
    """
    try:
        db = get_db()
        email = user_info['email']
        doc_ref = db.collection(USERS_COLLECTION).document(email)
        doc = doc_ref.get()
        
        now = datetime.utcnow().isoformat() + 'Z'
        
        if doc.exists:
            # Update last_login
            doc_ref.update({
                'last_login': now,
                'name': user_info.get('name', ''),
                'picture': user_info.get('picture', '')
            })
            return doc.to_dict()
        else:
            # Nuevo usuario — status pending
            new_user = {
                'email': email,
                'name': user_info.get('name', ''),
                'picture': user_info.get('picture', ''),
                'role': 'viewer',  # Default role
                'status': 'pending',  # Requiere aprobación admin
                'created_at': now,
                'last_login': now
            }
            doc_ref.set(new_user)
            print(f"📝 Nuevo usuario registrado: {email} (pending)")
            return new_user
    except Exception as e:
        print(f"❌ Error en create_or_update_user: {e}")
        return None

def check_user_authorized(email: str) -> tuple:
    """
    Verifica si un usuario está autorizado.
    Returns: (authorized: bool, user_data: dict, message: str)
    """
    user = get_user(email)
    
    if user is None:
        return False, None, 'Usuario no registrado'
    
    if user.get('status') == 'active':
        return True, user, 'OK'
    elif user.get('status') == 'pending':
        return False, user, 'Tu cuenta está pendiente de aprobación por un administrador.'
    elif user.get('status') == 'blocked':
        return False, user, 'Tu cuenta ha sido bloqueada. Contacta al administrador.'
    else:
        return False, user, f"Estado desconocido: {user.get('status')}"

def is_admin(email: str) -> bool:
    """Verifica si un usuario es admin."""
    user = get_user(email)
    return user is not None and user.get('role') == 'admin' and user.get('status') == 'active'

def list_all_users() -> list:
    """Lista todos los usuarios de cmarg_users."""
    try:
        db = get_db()
        docs = db.collection(USERS_COLLECTION).stream()
        users = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            users.append(data)
        return sorted(users, key=lambda x: x.get('created_at', ''), reverse=True)
    except Exception as e:
        print(f"❌ Error listando usuarios: {e}")
        return []

def update_user_status(email: str, status: str) -> bool:
    """Actualiza el status de un usuario (active/pending/blocked)."""
    try:
        db = get_db()
        db.collection(USERS_COLLECTION).document(email).update({
            'status': status,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        })
        print(f"✅ Usuario {email} → status: {status}")
        return True
    except Exception as e:
        print(f"❌ Error actualizando status: {e}")
        return False

def update_user_role(email: str, role: str) -> bool:
    """Actualiza el rol de un usuario (admin/analyst/viewer)."""
    try:
        db = get_db()
        db.collection(USERS_COLLECTION).document(email).update({
            'role': role,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        })
        print(f"✅ Usuario {email} → role: {role}")
        return True
    except Exception as e:
        print(f"❌ Error actualizando rol: {e}")
        return False

def delete_user(email: str) -> bool:
    """Elimina un usuario de Firestore."""
    try:
        db = get_db()
        db.collection(USERS_COLLECTION).document(email).delete()
        print(f"🗑️ Usuario {email} eliminado")
        return True
    except Exception as e:
        print(f"❌ Error eliminando usuario: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# Cookie Management
# ══════════════════════════════════════════════════════════════

COOKIE_NAME = 'cmarg_session'
COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 días

def set_session_cookie(response, token: str):
    """Establece la cookie de sesión."""
    response.set_cookie(
        COOKIE_NAME, 
        token, 
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite='Lax'
    )
    return response

def clear_session_cookie(response):
    """Elimina la cookie de sesión."""
    response.delete_cookie(COOKIE_NAME)
    return response

def get_session_token():
    """Obtiene el token de la cookie de sesión."""
    return request.cookies.get(COOKIE_NAME)


# ══════════════════════════════════════════════════════════════
# Middleware / Decorador
# ══════════════════════════════════════════════════════════════

def get_current_user():
    """
    Obtiene el usuario actual desde la cookie de sesión.
    Returns: (user_info, user_data) o (None, None)
    """
    token = get_session_token()
    if not token:
        return None, None
    
    user_info = verify_firebase_token(token)
    if not user_info:
        return None, None
    
    authorized, user_data, msg = check_user_authorized(user_info['email'])
    if not authorized:
        return user_info, None
    
    return user_info, user_data


# ══════════════════════════════════════════════════════════════
# Seedear admin inicial
# ══════════════════════════════════════════════════════════════

def seed_admin(email: str = 'cristian.avila@geologgia.cl'):
    """Crea el usuario admin inicial si no existe."""
    try:
        db = get_db()
        doc_ref = db.collection(USERS_COLLECTION).document(email)
        doc = doc_ref.get()
        
        if not doc.exists:
            doc_ref.set({
                'email': email,
                'name': 'Cristian Ávila',
                'role': 'admin',
                'status': 'active',
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'last_login': ''
            })
            print(f"✅ Admin inicial creado: {email}")
        else:
            data = doc.to_dict()
            if data.get('role') != 'admin':
                doc_ref.update({'role': 'admin', 'status': 'active'})
                print(f"✅ Usuario {email} promovido a admin")
            else:
                print(f"ℹ️ Admin {email} ya existe")
    except Exception as e:
        print(f"⚠️ No se pudo seedear admin: {e}")
