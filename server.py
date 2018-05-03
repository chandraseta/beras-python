from flask import Flask, request, redirect, json
from werkzeug.utils import secure_filename
import random
import os

UPLOAD_FOLDER = '.\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["GET","POST"])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No File Part"
        file = request.files['file']
        if file.filename == '':
            return "No File Selected"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) #TODO: Hash filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            #classify_image and return JSON
            
            return "Good"
    return '''
        <html>
        <head>
        <title> Classifier </title>
        </head>
        <body>
            <h1> Whoops! </h1>
            <p> You've seemed to access the wrong URL </p>
            <form method="post" enctype="multipart/form-data">
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
        </body>
        </html>
    '''