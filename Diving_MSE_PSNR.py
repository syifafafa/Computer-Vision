import cv2
import os
import numpy as np

# Fungsi untuk menghitung MSE
def calculate_mse(original, processed):
    mse = np.mean((original - processed) ** 2)
    return mse

# Fungsi untuk menghitung PSNR
def calculate_psnr(original, processed):
    psnr = cv2.PSNR(original, processed)
    return psnr

# Fungsi untuk memproses frame
def process_frame(frame, mse_values, psnr_values):
    # Konversi ke grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhancement 1: Histogram Equalization
    hist_eq_frame = cv2.equalizeHist(gray_frame)

    # Enhancement 2: Contrast Stretching
    min_val, max_val = np.min(gray_frame), np.max(gray_frame)
    contrast_stretch_frame = ((gray_frame - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Enhancement 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_frame = clahe.apply(gray_frame)

    # Median Filtering
    kernel_size = 5
    filtered_hist_eq = cv2.medianBlur(hist_eq_frame, kernel_size)
    filtered_contrast_stretch = cv2.medianBlur(contrast_stretch_frame, kernel_size)
    filtered_clahe = cv2.medianBlur(clahe_frame, kernel_size)

    # Evaluasi MSE
    mse_hist_eq = calculate_mse(gray_frame, filtered_hist_eq)
    mse_contrast_stretch = calculate_mse(gray_frame, filtered_contrast_stretch)
    mse_clahe = calculate_mse(gray_frame, filtered_clahe)

    # Evaluasi PSNR
    psnr_hist_eq = calculate_psnr(gray_frame, filtered_hist_eq)
    psnr_contrast_stretch = calculate_psnr(gray_frame, filtered_contrast_stretch)
    psnr_clahe = calculate_psnr(gray_frame, filtered_clahe)

    # Simpan hasil MSE dan PSNR
    mse_values['hist_eq'].append(mse_hist_eq)
    mse_values['contrast_stretch'].append(mse_contrast_stretch)
    mse_values['clahe'].append(mse_clahe)

    psnr_values['hist_eq'].append(psnr_hist_eq)
    psnr_values['contrast_stretch'].append(psnr_contrast_stretch)
    psnr_values['clahe'].append(psnr_clahe)

# Fungsi untuk memproses video
def process_video(input_folder_path, filename, mse_values_total, psnr_values_total):
    video_path = os.path.join(input_folder_path, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Tidak bisa membuka video {filename}")
        return

    # List untuk menyimpan MSE dan PSNR tiap frame
    mse_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}
    psnr_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Proses frame
        process_frame(frame, mse_values, psnr_values)

    cap.release()

    # Menghitung rata-rata MSE dan PSNR dari semua frame
    avg_mse_hist_eq = np.mean(mse_values['hist_eq'])
    avg_mse_contrast_stretch = np.mean(mse_values['contrast_stretch'])
    avg_mse_clahe = np.mean(mse_values['clahe'])

    avg_psnr_hist_eq = np.mean(psnr_values['hist_eq'])
    avg_psnr_contrast_stretch = np.mean(psnr_values['contrast_stretch'])
    avg_psnr_clahe = np.mean(psnr_values['clahe'])

    # Menyimpan hasil rata-rata MSE dan PSNR dari video ini ke total
    mse_values_total['hist_eq'].append(avg_mse_hist_eq)
    mse_values_total['contrast_stretch'].append(avg_mse_contrast_stretch)
    mse_values_total['clahe'].append(avg_mse_clahe)

    psnr_values_total['hist_eq'].append(avg_psnr_hist_eq)
    psnr_values_total['contrast_stretch'].append(avg_psnr_contrast_stretch)
    psnr_values_total['clahe'].append(avg_psnr_clahe)

    print(f"Rata-rata MSE dan PSNR untuk video {filename}:")
    print(f"  Histogram Equalization - MSE: {avg_mse_hist_eq:.2f}, PSNR: {avg_psnr_hist_eq:.2f} dB")
    print(f"  Contrast Stretching - MSE: {avg_mse_contrast_stretch:.2f}, PSNR: {avg_psnr_contrast_stretch:.2f} dB")
    print(f"  CLAHE - MSE: {avg_mse_clahe:.2f}, PSNR: {avg_psnr_clahe:.2f} dB")

# Input dan Output folder di Google Drive
input_folder_path = "D:\\7_smt.7\\ComVis\\UTS\\diving"
output_folder_path = "D:\\7_smt.7\\ComVis\\UTS\\Out"

# Membuat output folder jika belum ada
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# List untuk menyimpan total MSE dan PSNR semua video
mse_values_total = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}
psnr_values_total = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}

# Proses semua video dalam folder input
for filename in os.listdir(input_folder_path):
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        process_video(input_folder_path, filename, mse_values_total, psnr_values_total)

# Menghitung rata-rata total MSE dan PSNR untuk semua video
overall_avg_mse_hist_eq = np.mean(mse_values_total['hist_eq'])
overall_avg_mse_contrast_stretch = np.mean(mse_values_total['contrast_stretch'])
overall_avg_mse_clahe = np.mean(mse_values_total['clahe'])

overall_avg_psnr_hist_eq = np.mean(psnr_values_total['hist_eq'])
overall_avg_psnr_contrast_stretch = np.mean(psnr_values_total['contrast_stretch'])
overall_avg_psnr_clahe = np.mean(psnr_values_total['clahe'])

print("\nRata-rata MSE dan PSNR dari semua video:")
print(f"Histogram Equalization - MSE: {overall_avg_mse_hist_eq:.2f}, PSNR: {overall_avg_psnr_hist_eq:.2f} dB")
print(f"Contrast Stretching - MSE: {overall_avg_mse_contrast_stretch:.2f}, PSNR: {overall_avg_psnr_contrast_stretch:.2f} dB")
print(f"CLAHE - MSE: {overall_avg_mse_clahe:.2f}, PSNR: {overall_avg_psnr_clahe:.2f} dB")
