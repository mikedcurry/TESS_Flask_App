"""Main Application and routing Logic for TESS Flask App"""
from decouple import config
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from .models import DB

def create_app():
    """create and config an instance of the Flask App"""
    app = Flask(__name__)

    # configure DB, will need to update this when changing DBs?
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['ENV'] = config('ENV')
    DB.init_app(app)

    # Create home route
    @app.route('/')
    def root():
        return render_template('home.html', title = 'Home')
    

    return app
    