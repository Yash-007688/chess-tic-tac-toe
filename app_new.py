from flask import Flask, render_template, request, session, redirect, url_for  
import random  
import os  
  
app = Flask(__name__)  
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here') 
