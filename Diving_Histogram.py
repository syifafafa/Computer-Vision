import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Folder input dan output
input_folder = "D:\\7_smt.7\\ComVis\\UTS\\inputHist"  # Ganti dengan path folder input
output_folder = "D:\\7_smt.7\\ComVis\\UTS\\inputHist\\hist"  # Ganti dengan path folder output

# Membuat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Fungsi untuk menyimpan histogram
def save_histogram(image_path, output_path):
    # Baca gambar
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Jika gambar tidak ada, skip
    if img is None:
        print(f"Image {image_path} not found.")
        return

    # Hitung histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Buat grafik histogram
    plt.figure()
    plt.bar(np.arange(256), hist[:, 0], width=1)
    plt.xlim([0, 256])

    # Simpan grafik sebagai gambar
    plt.savefig(output_path)
    plt.close()

# Loop melalui semua file di folder input
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"hist_{filename}")
        save_histogram(input_path, output_path)
        print(f"Saved histogram for {filename}")

print("Proses selesai!")
