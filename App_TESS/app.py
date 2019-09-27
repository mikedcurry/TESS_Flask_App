"""Main Application and routing Logic for TESS Flask App"""
from decouple import config
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from .models import *
from .light_curve import *
from .Data_in import *
# from .predict import *

def create_app():
    """create and config an instance of the Flask App"""
    app = Flask(__name__)

    # configure DB, will need to update this when changing DBs?
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['ENV'] = config('ENV')
    DB.init_app(app)
    # with app.app_context():
    #     db.create_all()

    # Create home route
    @app.route('/')
    def root():
        #Pull example data from Notebooks folder. Will be be pulled from sql DB in the future.
        return render_template('home.html', 
                                title = 'Finding Planets:TESS', 
                                toi_table=(TOI_Table.query.all()), 
                                tic_table=(TIC_Cat_Table.query.all())
                               )     

    @app.route('/total_reset')
    def total_reset():
        DB.drop_all()
        DB.create_all()
        get_visual_data()
        get_toi_data()
        get_tic_catalog()
        return render_template('home.html', title='Reset Database!')

    @app.route('/predict')
    def predict():
        get_all_predictions()
        return render_template('home.html', title='prediction pipeline works!')

    # @app.route('/test')
    #     def get_urls(tic_id):
    #     urls = Visual_Table.query.filter_by(TIC_ID=tic_id).all()
    #     urls = [url.dataURL for url in urls]
    #     return urls

    return app
    