from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
import json


app = Flask(__name__)
# Muat data latih
with open('C:/skripsi/public/skripsi.json') as json_file:
    data_latih = json.load(json_file)['DataLatih']

# Konversi data latih ke format yang diperlukan
habit_data_latih = []
for key, value in data_latih.items():
    waktu = datetime.strptime(value['Waktu'], "%H:%M:%S")
    waktu_detik = waktu.hour * 3600 + waktu.minute * 60 + waktu.second
    habit_data_latih.append([
        waktu_detik,
        1 if value['dapur'] == 'hidup' else 0,
        1 if value['kamar'] == 'hidup' else 0,
        1 if value['kamar2'] == 'hidup' else 0,
        1 if value['ruangtamu'] == 'hidup' else 0,
        1 if value['teras'] == 'hidup' else 0,
        1 if value['toilet'] == 'hidup' else 0
    ])

habit_data_latih = np.array(habit_data_latih)

# Muat data uji
with open('C:/skripsi/public/skripsi.json') as json_file:
    data_uji = json.load(json_file)['DataUji']

predicted_data = {}
k_neighbors = 3

# Fungsi untuk memprediksi keadaan lampu berdasarkan waktu terdekat
def euclidean_distance(x1, x2):
    return np.abs(x1 - x2)

def predict_lights_by_time(time, k, habit_data):
    waktu = datetime.strptime(time, "%H:%M:%S")
    waktu_detik = waktu.hour * 3600 + waktu.minute * 60 + waktu.second
    habits = np.array([
        waktu_detik,
        *[1 if value == 'hidup' else 0 for value in data_latih['-NUbhKQusYWmh4qcwaEW'].values()]
    ])

    # Menghitung jarak Euclidean hanya berdasarkan waktu dengan semua data dalam habit_data
    distances = [euclidean_distance(habits[0], habit[0]) for habit in habit_data]

    # Mengambil indeks k tetangga terdekat
    k_indices = np.argsort(distances)[:k]

    # Mengambil status lampu dari k tetangga terdekat
    neighbor_lights = habit_data[k_indices][:, 1:]

    n_neighbors = len(k_indices)
    predicted_lights = []
    for lamp in neighbor_lights.T:
        lamp_state = 1 if np.sum(lamp) > n_neighbors / 2 else 0
        predicted_lights.append(lamp_state)

    return predicted_lights

# Prediksi keadaan lampu untuk data uji
for key, value in data_uji.items():
    waktu = value['Waktu']
    predicted_lights = predict_lights_by_time(waktu, k_neighbors, habit_data_latih)
    predicted_data[key] = predicted_lights

@app.route('/lamp_status', methods=['GET'])
def get_lamp_status():
    waktu_terbaru = datetime.now()
    formatted_time = waktu_terbaru.strftime("%H:%M:%S")
    k_neighbors = 3
    predicted_lights = predict_lights_by_time(formatted_time, k_neighbors, habit_data_latih)

    status_strings = ["hidup" if status == 1 else "mati" for status in predicted_lights]

    response = {
        "lamp_status": status_strings
    }

    return jsonify(response)


@app.route('/accuracy', methods=['GET'])
def calculate_accuracy():
    try:
        confusion_matrix = {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
        }

        for key, value in data_uji.items():
            actual_lights = [
                1 if value['dapur'] == 'hidup' else 0,
                1 if value['kamar'] == 'hidup' else 0,
                1 if value['kamar2'] == 'hidup' else 0,
                1 if value['ruangtamu'] == 'hidup' else 0,
                1 if value['teras'] == 'hidup' else 0,
                1 if value['toilet'] == 'hidup' else 0
            ]
            predicted_lights = predicted_data[key]

            for i in range(len(actual_lights)):
                if actual_lights[i] == 1 and predicted_lights[i] == 1:
                    confusion_matrix['true_positive'] += 1
                elif actual_lights[i] == 0 and predicted_lights[i] == 1:
                    confusion_matrix['false_positive'] += 1
                elif actual_lights[i] == 0 and predicted_lights[i] == 0:
                    confusion_matrix['true_negative'] += 1
                elif actual_lights[i] == 1 and predicted_lights[i] == 0:
                    confusion_matrix['false_negative'] += 1

        accuracy = (confusion_matrix['true_positive'] + confusion_matrix['true_negative']) / (
                    confusion_matrix['true_positive'] + confusion_matrix['false_positive'] +
                    confusion_matrix['true_negative'] + confusion_matrix['false_negative'])

        response = {
        "akurasi": accuracy * 100
    }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
