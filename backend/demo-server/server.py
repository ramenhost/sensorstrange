from flask import Flask, request
import os
import zipfile
from LanguageModel import LanguageModel
from infer import pipeline
import threading
import tensorflow as tf

UPLOAD_FOLDER = 'uploads/'
DATA_FOLDER = 'data/'
ALLOWED_EXTENSIONS = set(['zip'])

print('-------------------------------------')
print('Text Inference from Smartphone motion')
print('-------------------------------------')

def init():
  global lm, train_ses, graph
  lm = LanguageModel('corpus/google-10000-english.txt')
  lm.load_weights('model/keras_char_rnn.500.h5')
  graph = tf.get_default_graph()
  train_ses = ''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def handle(ses):
  global train_ses, lm
  if train_ses == '':
    train_ses = ses
    return
  with graph.as_default():
    print(pipeline(train_ses, ses, lm))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/upload", methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename);
            file.save(file_path)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(DATA_FOLDER)
            zip_ref.close()
            t = threading.Thread(target=handle, args=(file.filename.split('.')[0],))
            t.start()
            return 'starting inference'

if __name__ == "__main__":
  print('Listening for connections on 8000')
  init()
  app.run(host='0.0.0.0', port=8000, debug=True)