# This is where our intial database configuration will be written
from flask_sqlalchemy import SQLAlchemy

DB = SQLAlchemy()

#create <table> class
# class User(DB.Model):
#     id = DB.Column(DB.Integer, primary_key=True)
#     # Example Columns below
#     # name = DB.Column(DB.String(15), nullable=False)
#     # newest_tweet_id = DB.Column(DB.BigInteger)
#     def __repr__(self):
#         return '<User {}>'.format(self.name)