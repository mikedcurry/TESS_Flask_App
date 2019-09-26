import pickle
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .models import DB, TOI_Table, TIC_Cat_Table
from sqlalchemy.inspection import inspect

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

# Shaping the data for model training:
def shape_data(df):
    df = get_data()

    df['TFOPWG_Disposition'] = df[
        'TFOPWG_Disposition'].replace({'KP': 1, 'CP': 1, 'FP': 0})

    # Creating confirmed planets dataframe:
    cp_df = df[df['TFOPWG_Disposition'] == 1]

    # Creating false positives dataframe:
    fp_df = df[df['TFOPWG_Disposition'] == 0]

    # Train/test split on both dataframes:
    cp_train, cp_test = train_test_split(cp_df, random_state=42)
    fp_train, fp_test = train_test_split(fp_df, random_state=42)

    # Combining training dataframes:
    train = cp_train.append(fp_train)
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combining test dataframes:
    test = cp_test.append(fp_test)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Dropping columns from training dataframe that aren't used in model:
    X_train = train.drop(columns=['TIC_ID', 'TOI', 'TFOPWG_Disposition'])
    # Getting labels for training data:
    y_train = train['TFOPWG_Disposition'].astype(int)

    # Dropping columns from test dataframe that aren't used in model:
    X_test = test.drop(columns=['TIC_ID', 'TOI', 'TFOPWG_Disposition'])
    # Getting labels for test data:
    y_test = test['TFOPWG_Disposition'].astype(int)

    return X_train, y_train, X_test, y_test

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

# Creating pipeline:
def train_model():
    
    df = get_data()

    X_train, y_train, X_test, y_test = shape_data(df)
    
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        RobustScaler(),
        KerasClassifier(build_fn=create_model, verbose=2)
        )

    pipeline.fit(X_train, y_train,
             kerasclassifier__batch_size=5,
             kerasclassifier__epochs=50,
             kerasclassifier__validation_split=.2,
             kerasclassifier__verbose=2)

    print('\n\n')
    print(f'Train Accuracy Score:', pipeline.score(X_train, y_train), '\n')
    print(f'Test Accuracy Score:', pipeline.score(X_test, y_test))

    return pipeline

def save_model():
    pipeline = train_model()
    pipeline.named_steps['kerasclassifier'].model.save('keras_classifier.h5')
    pipeline.named_steps['kerasclassifier'].model = None
    pickle.dump(pipeline, open('tess_pipeline.pkl', 'wb'))

save_model()