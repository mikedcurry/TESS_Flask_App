# This is where our intial database configuration will be written
from flask_sqlalchemy import SQLAlchemy

DB = SQLAlchemy()

# create visualization info class
class Visual_Table(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    tic_id = DB.Column(DB.BigInteger, nullable=False)
    data_url = DB.Column(DB.String(100))
    def __repr__(self):
        return '(TIC_ID %r, url %r)' %(self.tic_id, self.data_url)

# class 