"""Entry point for TESS planet finding Flask App"""
from .app import create_app

APP = create_app()
# APP.app_context().push()