# This is where our intial database configuration will be written
from flask_sqlalchemy import SQLAlchemy

DB = SQLAlchemy()

# create visualization info class
class Visual_Table(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    TIC_ID = DB.Column(DB.BigInteger, nullable=False)
    dataURL = DB.Column(DB.String(100))
    def __repr__(self):
        return '<Visual_Table {}>'.format(self.TIC_ID)

class TOI_Table(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    TIC_ID = DB.Column(DB.BigInteger, nullable=False)
    TOI = DB.Column(DB.Float)
    Epoch = DB.Column(DB.Float)
    Period = DB.Column(DB.Float)
    Duration = DB.Column(DB.Float)
    Depth = DB.Column(DB.Float)
    Planet_Radius = DB.Column(DB.Float)
    Planet_Insolation = DB.Column(DB.Float)
    Planet_Equil_Temp = DB.Column(DB.Float)
    Planet_SNR = DB.Column(DB.Float)   
    Stellar_Distance = DB.Column(DB.Float)
    Stellar_log_g = DB.Column(DB.Float)
    Stellar_Radius = DB.Column(DB.Float)
    TFOPWG_Disposition = DB.Column(DB.String)
    def __repr__(self):
        return '<TOI_Table {}>'.format(self.TIC_ID)


class TIC_Cat_Table(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    TIC_ID = DB.Column(DB.BigInteger, nullable=False)
    ra = DB.Column(DB.Float)
    dec = DB.Column(DB.Float)
    pmRA = DB.Column(DB.Float)
    pmDEC = DB.Column(DB.Float)
    plx = DB.Column(DB.Float)
    gallong = DB.Column(DB.Float)
    gallat = DB.Column(DB.Float)
    eclong = DB.Column(DB.Float)
    eclat = DB.Column(DB.Float)
    Tmag = DB.Column(DB.Float)
    Teff = DB.Column(DB.Float)
    logg = DB.Column(DB.Float)
    MH = DB.Column(DB.Float)
    rad = DB.Column(DB.Float)
    mass = DB.Column(DB.Float)
    rho = DB.Column(DB.Float)
    lum = DB.Column(DB.Float)
    d = DB.Column(DB.Float)
    ebv = DB.Column(DB.Float)
    numcont = DB.Column(DB.Float)
    contratio = DB.Column(DB.Float)
    priority = DB.Column(DB.Float)
    def __repr__(self):
        return '<TIC_Cat_Table {}>'.format(self.TIC_ID)


    # tic_id = DB.Column(DB.BigInteger, nullable=False)
    # data_url = DB.Column(DB.String(100))
    # def __repr__(self):
    #     return '(TIC_ID %r, url %r)' %(self.tic_id, self.data_url)

# class 
