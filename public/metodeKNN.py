import json
import numpy as np
from datetime import datetime

# Membaca data dari file JSON
with open('C:/skripsi/public/2minggu.json') as json_file:
    data = json.load(json_file)

# Mengubah data waktu menjadi detik sejak tengah malam
habit_data = []
for key, value in data['Data'].items():
    waktu = datetime.strptime(value['Waktu'], "%H:%M:%S")
    waktu_detik = waktu.hour * 3600 + waktu.minute * 60 + waktu.second
    habit_data.append([
        waktu_detik,
        1 if value['dapur'] == 'hidup' else 0,
        1 if value['kamarUtama'] == 'hidup' else 0,
        1 if value['kamarKedua'] == 'hidup' else 0,
        1 if value['ruangTamu'] == 'hidup' else 0,
        1 if value['teras'] == 'hidup' else 0,
        1 if value['toilet'] == 'hidup' else 0
    ])

habit_data = np.array(habit_data)

# Fungsi untuk menghitung jarak Euclidean hanya berdasarkan waktu
def euclidean_distance(x1, x2):
    return np.abs(x1 - x2)

# Fungsi untuk memprediksi keadaan lampu berdasarkan waktu terdekat
def predict_lights_by_time(time, k):
    waktu = datetime.strptime(time, "%H:%M:%S")
    waktu_detik = waktu.hour * 3600 + waktu.minute * 60 + waktu.second

    # Menghitung jarak Euclidean hanya berdasarkan waktu dengan semua data dalam habit_data
    distances = [euclidean_distance(waktu_detik, habit[0]) for habit in habit_data]

    # Mengambil indeks k tetangga terdekat
    k_indices = np.argsort(distances)[:k]

    # Mengambil status lampu dari k tetangga terdekat
    neighbor_lights = habit_data[k_indices][:, 1:]

    n_neighbors = len(k_indices)
    predicted_lights = []
    for lamp in neighbor_lights.T:
        lamp_state = 'hidup' if np.sum(lamp) > n_neighbors / 2 else 'mati'
        predicted_lights.append(lamp_state)

    return predicted_lights

# Masukkan waktu baru di sini
new_time = "06:30:00"  # Contoh waktu baru
k_neighbors = 3

predicted_lights = predict_lights_by_time(new_time, k_neighbors)
print("Hasil prediksi keadaan lampu:")
for lamp_name, lamp_state in zip(data['Data']['-NUbhKQusYWmh4qcwaEW'].keys(), predicted_lights):
    print(f"{lamp_name}: {lamp_state}")