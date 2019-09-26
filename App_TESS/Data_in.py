import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from astroquery.mast import Catalogs, Observations
from astropy.table import Table
# from tqdm import tqdm                       # Will need to remove this througout
from .models import *


# fetch TIC IDs from caltech
def get_visual_data():
    try:
        # Getting labelled TESS Objects of Interest dataframe from Caltech:
        toi = pd.read_csv('https://exofop.ipac.caltech.edu/tess/' + 
                  'download_toi.php?sort=toi&output=csv')
        print('got past csv import')
        # Isolating TIC IDs and TFOPWG Disposition values to use as target:
        toi = toi[['TIC ID', 'TFOPWG Disposition']]
        toi = toi.rename(columns={'TIC ID': 'TIC_ID'})

        # Getting additional data on TESS Objects of Interest from STScI:
        tic_catalog = pd.DataFrame()
        for tic_id in toi['TIC_ID'].unique():
            row_data = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
            row_data = row_data.to_pandas()
            tic_catalog = tic_catalog.append(row_data)
        tic_catalog = tic_catalog.reset_index(drop=True)
        # Renaming ID column to make this consistent with Caltech TOI dataframe:
        tic_catalog = tic_catalog.rename(columns={'ID': 'TIC_ID'})
        print('got past merging toi with tic_cat')

        # Getting all dataproducts for TESS Objects of Interest from STScI:
        dataproducts = pd.DataFrame()
        for tic_id in toi['TIC_ID']:                
            row_data = Observations.query_criteria(obs_collection="TESS",
                                                target_name=tic_id)
            row_data = row_data.to_pandas()
            dataproducts = dataproducts.append(row_data)
        dataproducts = dataproducts.reset_index(drop=True)
        # Isolating TIC IDs (target_name) and dataURL values to get associated files:
        dataproducts = dataproducts[['target_name', 'dataURL']]
        # Renaming ID column to make this consistent with Caltech TOI dataframe:
        dataproducts = dataproducts.rename(columns={'target_name': 'TIC_ID'})

    except Exception as e:
        print('Error importing data: ')
        raise e
    for index, row in dataproducts.iterrows():
        # print(row[0], row[1])
        new = Visual_Table(TIC_ID=row[0], dataURL=row[1])
        DB.session.add(new)
        DB.session.commit()
    return


def get_toi_data():
    try: 
        # Getting labelled TESS Objects of Interest dataframe from Caltech:
        toi = pd.read_csv('https://exofop.ipac.caltech.edu/tess/' + 
                    'download_toi.php?sort=toi&output=csv')
    except Exception as e:
        print('failed to import initial csv from caltech') 
        raise e
    try:
        # Isolating columns we want:
        toi = toi[['TIC ID',
            'TOI',
            'Epoch (BJD)',
            'Period (days)',
            'Duration (hours)',
            'Depth (mmag)',
            'Planet Radius (R_Earth)',
            'Planet Insolation (Earth Flux)',
            'Planet Equil Temp (K)',
            'Planet SNR',
            'Stellar Distance (pc)',
            'Stellar log(g) (cm/s^2)',
            'Stellar Radius (R_Sun)',
            'TFOPWG Disposition',
            ]]
        toi.columns = toi.columns.str.replace(' ', '_')
    except:
        print('failed to filter df')
    for index, row in toi.iterrows():
        new = TOI_Table(TIC_ID=row[0], 
                           TOI=row[1],
                           Epoch=row[2],
                           Period=row[3],
                           Duration=row[4],
                           Depth=row[5],
                           Planet_Radius=row[6],
                           Planet_Insolation=row[7],
                           Planet_Equil_Temp=row[8],
                           Planet_SNR=row[9],
                           Stellar_Distance=row[10],
                           Stellar_log_g=row[11],
                           Stellar_Radius=row[12],
                           TFOPWG_Disposition=row[13]
                           )
        DB.session.add(new)
        DB.session.commit()
    return


def get_tic_catalog():
    toi = pd.read_csv('https://exofop.ipac.caltech.edu/tess/' + 
                    'download_toi.php?sort=toi&output=csv')
    toi = toi[['TIC ID',
            'TOI',
            'Epoch (BJD)',
            'Period (days)',
            'Duration (hours)',
            'Depth (mmag)',
            'Planet Radius (R_Earth)',
            'Planet Insolation (Earth Flux)',
            'Planet Equil Temp (K)',
            'Planet SNR',
            'Stellar Distance (pc)',
            'Stellar log(g) (cm/s^2)',
            'Stellar Radius (R_Sun)',
            'TFOPWG Disposition',
            ]]
    toi.columns = toi.columns.str.replace(' ', '_')
    try:
        tic_catalog = pd.DataFrame()
        for tic_id in toi['TIC_ID'].unique():
            row_data = Catalogs.query_criteria(catalog="Tic", ID=tic_id)
            row_data = row_data.to_pandas()
            tic_catalog = tic_catalog.append(row_data)
        tic_catalog = tic_catalog.reset_index(drop=True)
    except:
        print('failed to import')
    try:
        # Renaming ID column to make this consistent with Caltech TOI dataframe:
        tic_catalog = tic_catalog.rename(columns={'ID': 'TIC ID'})

        # Isolating columns we want:
        tic_catalog = tic_catalog[['TIC ID',
                                'ra',
                                'dec',
                                'pmRA',
                                'pmDEC',
                                'plx',
                                'gallong',
                                'gallat',
                                'eclong',
                                'eclat',
                                'Tmag',
                                'Teff',
                                'logg',
                                'MH',
                                'rad',
                                'mass',
                                'rho',
                                'lum',
                                'd',
                                'ebv',
                                'numcont',
                                'contratio',
                                'priority']]
        tic_catalog.columns = tic_catalog.columns.str.replace(' ', '_')
    except:
            print('failed to filter columns')
    for index, row in tic_catalog.iterrows():
        new = TIC_Cat_Table(TIC_ID=row[0],
                           ra=row[1],
                           dec=row[2],
                           pmRA=row[3],
                           pmDEC=row[4],
                           plx=row[5],
                           gallong=row[6],
                           gallat=row[7],
                           eclong=row[8],
                           eclat=row[9],
                           Tmag=row[10],
                           Teff=row[11],
                           logg=row[12],
                           MH=row[13],
                           rad=row[14],
                           mass=row[15],
                           rho=row[16],
                           lum=row[17],
                           d=row[18],
                           ebv=row[19],
                           numcont=row[20],
                           contratio=row[21],
                           priority=row[22]
                          )
        DB.session.add(new)
        DB.session.commit()
    return

def toi_df():
    df = pd.DataFrame()
    rows = TOI_Table.query.all()

    
    return df