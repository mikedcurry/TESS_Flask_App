import pandas as pd
from astroquery.mast import Catalogs, Observations
from astropy.table import Table
from tqdm import tqdm
from .models import DB, Visual_Table

# Getting labelled TESS Objects of Interest dataframe from Caltech:


# fetch TIC IDs from caltech
def get_data():
    try:
        toi = pd.read_csv('https://exofop.ipac.caltech.edu/tess/' + 
                  'download_toi.php?sort=toi&output=csv')
        dataproducts = pd.DataFrame()
        for tic_id in tqdm(toi['TIC ID']):
            row_data = Observations.query_criteria(obs_collection="TESS",
                                           target_name=tic_id)
            row_data = row_data.to_pandas()
            dataproducts = dataproducts.append(row_data)
        dataproducts = dataproducts.reset_index(drop=True)
        useful_data = dataproducts[['target_name', 'target_name']]
        # Not sure if the below will work...
        for row in useful_data:
            pair = Visual_Table(tic_id = useful_data['target_name']
                                , data_url = useful_data['target_name'])
            DB.session.add(pair)
    except:
        print('Error importing data')
        raise e
    else:
        DB.session.commit()
