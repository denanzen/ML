from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  # Untuk mengatasi masalah CORS
import joblib
import numpy as np
import logging

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS

# Konfigurasi database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data_hutan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Logger
logging.basicConfig(level=logging.DEBUG)

# Load model
try:
    model = joblib.load('model_kesehatan_hutan.pkl')
except FileNotFoundError:
    logging.error("File model_kesehatan_hutan.pkl tidak ditemukan.")
    model = None

# Model untuk tabel Data Hutan
class DataHutan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kesehatan_pohon = db.Column(db.Integer)
    kondisi_tanah = db.Column(db.Integer)
    kualitas_air = db.Column(db.Integer)
    tingkat_gangguan = db.Column(db.Integer)
    prediction = db.Column(db.String(20))

# Inisialisasi database
with app.app_context():
    db.create_all()

@app.route('/')
def awal():
    return render_template('awal.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/beranda')
def beranda():
    return render_template('beranda.html')

@app.route('/utama')
def utama():
    return render_template('utama.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model tidak tersedia."}), 500

    data = request.json

    # Validasi input
    if not data or not all(key in data for key in ['kesehatan_pohon', 'kondisi_tanah', 'kualitas_air', 'tingkat_gangguan']):
        return jsonify({"error": "Data input tidak lengkap."}), 400

    try:
        # Ambil data dari input
        features = [
            int(data['kesehatan_pohon']),
            int(data['kondisi_tanah']),
            int(data['kualitas_air']),
            int(data['tingkat_gangguan'])
        ]
        features = np.array(features).reshape(1, -1)

        # Prediksi
        prediction = model.predict(features)[0]

        # Simpan data ke database
        new_data = DataHutan(
            kesehatan_pohon=data['kesehatan_pohon'],
            kondisi_tanah=data['kondisi_tanah'],
            kualitas_air=data['kualitas_air'],
            tingkat_gangguan=data['tingkat_gangguan'],
            prediction=prediction
        )
        db.session.add(new_data)
        db.session.commit()

        return jsonify({
            "input": data,
            "prediction": prediction
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {str(e)}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
