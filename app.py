from flask import Flask, render_template, request, session
import torch
import pandas as pd
import os
import csv
import chardet 
from test_model import predict, predict_in_file
from Neural_Architecture import LSTM_architecture
from math import ceil
from werkzeug.utils import secure_filename
#from fileinput import filename

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '0eb7dfc3048b40079b6b082ed4fb0794'

vocab_size = 111796
output_size = 1
embedding_dim = 128
hidden_dim = 128
number_of_layers = 2
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
model.load_state_dict(torch.load("model", map_location=torch.device('cpu')))
seq_length = 50
right_enc = 'UTF-8'

def detect_encoding(file_path): 
    with open(file_path, 'rb') as file: 
        detector = chardet.universaldetector.UniversalDetector() 
        for line in file: 
            detector.feed(line) 
            if detector.done: 
                break
        detector.close() 
    return detector.result['encoding']

def check(file_path, flag_size):
    encoding = detect_encoding(file_path)
    if right_enc in encoding:
        try:
            df = pd.read_csv(file_path)  #read the CSV file
 
            if df.empty:  # Check if the DataFrame is empty
                flag_size = True

        except pd.errors.EmptyDataError:
            flag_size = True

    return flag_size


@app.route('/', methods=["GET", "POST"])
def index():    
    return render_template('home.html')

@app.route('/upload', methods=["GET", "POST"])
def result():
    if request.method == 'POST':
        data_file_path = session.get('uploaded_data_file_path', None)

        uploaded_df = pd.read_csv(data_file_path, sep=';', header = None, names = ['Отзыв'])
        count_rows = len(uploaded_df)
        
        new_sentiment_values, new_prob_values = predict_in_file(uploaded_df, count_rows, model, seq_length)
        
        uploaded_df['Эмоциональная окраска'] = new_sentiment_values
        uploaded_df['Вероятность'] = new_prob_values
 
        uploaded_df_html = uploaded_df.to_html()
    return render_template('download.html', data_var = uploaded_df_html)

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    flag = False
    flag_file = False
    flag_enc = False
    flag_input = False
    flag_size = False
    flag_empty = False
    type_of_tonal = ""
    prob = 0
    name = ""
    if request.method == 'POST':
        if request.form.get('link_button'):
            flag = False
        elif request.form.get('submit_button'):
            name1 = request.form['text_tonal']
            if len(name1) != 0:
                flag = True
                name = name1
                type_of_tonal, pos_prob = predict(model, name, seq_length)
                if type_of_tonal == "Негативное сообщение":
                    prob = ceil((1 - pos_prob)*100)
                else:
                    prob = ceil(pos_prob*100)
            else:
                flag_input = True
        elif request.form.get('upload_button'):       
            uploaded_df = request.files['uploaded_file']
            data_filename = secure_filename(uploaded_df.filename)
            if (data_filename == ''):
                flag_empty = True
            else: 
                uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
                session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            
                data_file_path = session.get('uploaded_data_file_path', None)
                flag_size = check(data_file_path, flag_size)

                if not flag_size:
                    encoding = detect_encoding(data_file_path)
                    if right_enc in encoding:
                        flag_file = True
                    else:
                        flag_enc = True

    return render_template('senti.html', flag = flag, type_of_tonal = type_of_tonal, percent = "{} %".format(prob), text = name, flag_file = flag_file, flag_enc = flag_enc, flag_input = flag_input, flag_size = flag_size, flag_empty = flag_empty)

if __name__ == "__main__":
    app.run()


