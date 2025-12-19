import math
from collections import Counter

# ==============================================================
# MODUL NATIVE KNN (BUATAN SENDIRI)
# File ini berisi rumus matematika manual tanpa Scikit-Learn
# ==============================================================

def hitung_jarak_euclidean(data1, data2):
    """
    Fungsi untuk menghitung jarak lurus (Euclidean Distance)
    Rumus: Akar( (x1-y1)^2 + (x2-y2)^2 + ... )
    """
    distance = 0.0
    # Loop sebanyak jumlah fitur (Penghasilan, Tanggungan, dll)
    for i in range(len(data1)):
        # Hitung selisih kuadrat
        distance += (data1[i] - data2[i])**2
    
    # Akar kuadrat hasil penjumlahan
    return math.sqrt(distance)

def dapatkan_tetangga(training_set, test_row, k):
    """
    Fungsi untuk mencari K tetangga terdekat.
    Langkah:
    1. Hitung jarak data baru ke SEMUA data latih.
    2. Urutkan dari jarak terkecil.
    3. Ambil K data teratas.
    """
    distances = []
    
    # 1. Bandingkan data test dengan SEMUA data database
    for train_row in training_set:
        # train_row[:-1] artinya ambil fiturnya saja, label status jangan dihitung jaraknya
        dist = hitung_jarak_euclidean(test_row, train_row[:-1])
        distances.append((train_row, dist))
    
    # 2. Urutkan dari jarak terkecil (Ascending)
    distances.sort(key=lambda x: x[1])
    
    # 3. Ambil sebanyak K tetangga teratas
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
        
    return neighbors

def prediksi_native(training_set, test_row, k):
    """
    Fungsi Utama untuk menentukan keputusan (Layak/Tidak).
    Menggunakan konsep Voting Mayoritas.
    """
    # 1. Cari siapa tetangganya
    tetangga = dapatkan_tetangga(training_set, test_row, k)
    
    # 2. Ambil status (Label) dari tetangga tersebut
    # (Label ada di kolom terakhir atau indeks -1)
    output_values = [row[-1] for row in tetangga]
    
    # 3. Cari Modus (Nilai yang paling sering muncul)
    # Contoh: Jika tetangganya [1, 1, 1, 0, 0] -> Maka hasilnya 1 (Layak)
    # most_common(1) mengembalikan list tuple, misal [(1, 3)]
    suara_terbanyak = Counter(output_values).most_common(1)[0][0]
    
    return suara_terbanyak