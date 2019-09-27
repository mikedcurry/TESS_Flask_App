"""Main Application and routing Logic for TESS Flask App"""
from decouple import config
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from .models import DB, TOI_Table, TIC_Cat_Table
from .models import *
from .light_curve import *
from .Data_in import *

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
        #Pull example data from Notebooks folder. Will be be pulled from sql DB in the future.
        return render_template('home.html', 
                                title = 'Findin Planets:TESS', 
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
        # Gathering the necessary data from sql database:
        def get_data():
            # Pulling data from sql database
            toi_rows = TOI_Table.query.all()
            tic_catalog_rows = TIC_Cat_Table.query.all()
            toi_dict = {'TIC_ID': [row.TIC_ID for row in toi_rows],
                        'TOI': [row.TOI for row in toi_rows],
                        'Epoch': [row.Epoch for row in toi_rows],
                        'Period': [row.Period for row in toi_rows],
                        'Duration': [row.Duration for row in toi_rows],
                        'Depth': [row.Depth for row in toi_rows],
                        'Planet_Radius': [row.Planet_Radius for row in toi_rows],
                        'Planet_Insolation': [row.Planet_Insolation for row in toi_rows],
                        'Planet_Equil_Temp': [row.Planet_Equil_Temp for row in toi_rows],
                        'Planet_SNR': [row.Planet_SNR for row in toi_rows],
                        'Stellar_Distance': [row.Stellar_Distance for row in toi_rows],
                        'Stellar_log_g': [row.Stellar_log_g for row in toi_rows],
                        'Stellar_Radius': [row.Stellar_Radius for row in toi_rows],
                        'TFOPWG_Disposition': [row.TFOPWG_Disposition for row in toi_rows]}
            tic_catalog_dict = {'TIC_ID': [row.TIC_ID for row in tic_catalog_rows],
                                'ra': [row.ra for row in tic_catalog_rows],
                                'dec': [row.dec for row in tic_catalog_rows],
                                'pmRA': [row.pmRA for row in tic_catalog_rows],
                                'pmDEC': [row.pmDEC for row in tic_catalog_rows],
                                'plx': [row.plx for row in tic_catalog_rows],
                                'gallong': [row.gallong for row in tic_catalog_rows],
                                'gallat': [row.gallat for row in tic_catalog_rows],
                                'eclong': [row.eclong for row in tic_catalog_rows],
                                'eclat': [row.eclat for row in tic_catalog_rows],
                                'Tmag': [row.Tmag for row in tic_catalog_rows],
                                'Teff': [row.Teff for row in tic_catalog_rows],
                                'logg': [row.logg for row in tic_catalog_rows],
                                'MH': [row.MH for row in tic_catalog_rows],
                                'rad': [row.rad for row in tic_catalog_rows],
                                'mass': [row.mass for row in tic_catalog_rows],
                                'rho': [row.rho for row in tic_catalog_rows],
                                'lum': [row.lum for row in tic_catalog_rows],
                                'd': [row.d for row in tic_catalog_rows],
                                'ebv': [row.ebv for row in tic_catalog_rows],
                                'numcont': [row.numcont for row in tic_catalog_rows],                
                                'contratio': [row.contratio for row in tic_catalog_rows],
                                'priority': [row.priority for row in tic_catalog_rows]}
            toi = pd.DataFrame(toi_dict)
            tic_catalog = pd.DataFrame(tic_catalog_dict)

            df = toi.merge(tic_catalog, on='TIC_ID')
            return df

        # Shaping the data for input:
        def shape_data(df):
            # Dropping data not needed for model:
            X = df.drop(columns=['TIC_ID', 'TOI', 'TFOPWG_Disposition'])
            return X

        # Setting up model architecture for neural net:
        def create_model():
            # Instantiate model:
            model = Sequential()
            # Add input layer:
            model.add(Dense(20, input_dim=33, activation='relu'))
            # Add hidden layer:
            model.add(Dense(20, activation='relu'))
            # Add output layer:
            model.add(Dense(1, activation='sigmoid'))
            # Compile model:
            model.compile(loss='binary_crossentropy', optimizer='adam',
                    metrics=['accuracy'])
            return model

        # Loading pipeline:
        def load_pipeline():
            tess_pipeline = pickle.load(open('tess_pipeline.pkl', 'rb'))
            tess_pipeline.named_steps[
                'kerasclassifier'].model = load_model(
                    'keras_classifier.h5')
            return tess_pipeline

        # Get predictions for all observations:
        def get_all_predictions():
            y_pred_proba_full = load_pipeline().predict_proba(shape_data(get_data()))
            toi_index = pd.DataFrame(get_data()['TOI'])
            output_df = pd.DataFrame(y_pred_proba_full, columns=[
                'actual_exoplanet_prob', 'false_positive_prob'])
            output_df = toi_index.join(output_df)
            output_df['prediction'] = np.where(
                output_df['actual_exoplanet_prob'] >= output_df[
                    'false_positive_prob'],
                    'Actual Exoplanet', 'False Positive')
            output_df['prediction_prob'] = np.where(output_df[
                'actual_exoplanet_prob']>= output_df[
                    'false_positive_prob'], output_df[
                        'actual_exoplanet_prob'], output_df[
                            'false_positive_prob'])
            return output_df
            
        get_all_predictions()
        return render_template('home.html', title='prediction pipeline works!')

    # @app.route('/test')
    #     def get_urls(tic_id):
    #     urls = Visual_Table.query.filter_by(TIC_ID=tic_id).all()
    #     urls = [url.dataURL for url in urls]
    #     return urls

    return app
    