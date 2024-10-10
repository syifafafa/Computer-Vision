import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menghitung PSNR
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:  # Jika tidak ada perbedaan antara gambar asli dan diproses
        return float('inf'), float('inf')  # PSNR tak terhingga
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return mse, psnr

# Fungsi untuk memproses frame
def process_frame(frame, frame_count, video_output_folder, mse_values, psnr_values):
    # Konversi ke grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhancement 1: Histogram Equalization
    hist_eq_frame = cv2.equalizeHist(gray_frame)

    # Enhancement 2: Contrast Stretching
    p2, p98 = np.percentile(gray_frame, (2, 98))
    contrast_stretch_frame = cv2.normalize(gray_frame, None, alpha=p2, beta=p98, norm_type=cv2.NORM_MINMAX)

    # Enhancement 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_frame = clahe.apply(gray_frame)

    # Median Filtering
    kernel_size = 5
    filtered_hist_eq = cv2.medianBlur(hist_eq_frame, kernel_size)
    filtered_contrast_stretch = cv2.medianBlur(contrast_stretch_frame, kernel_size)
    filtered_clahe = cv2.medianBlur(clahe_frame, kernel_size)

    # Evaluasi MSE dan PSNR
    mse_hist_eq, psnr_hist_eq = calculate_psnr(gray_frame, filtered_hist_eq)
    mse_contrast_stretch, psnr_contrast_stretch = calculate_psnr(gray_frame, filtered_contrast_stretch)
    mse_clahe, psnr_clahe = calculate_psnr(gray_frame, filtered_clahe)

    # Simpan hasil MSE dan PSNR
    mse_values['hist_eq'].append(mse_hist_eq)
    mse_values['contrast_stretch'].append(mse_contrast_stretch)
    mse_values['clahe'].append(mse_clahe)

    psnr_values['hist_eq'].append(psnr_hist_eq)
    psnr_values['contrast_stretch'].append(psnr_contrast_stretch)
    psnr_values['clahe'].append(psnr_clahe)

    # Buat folder khusus untuk frame
    frame_output_folder = os.path.join(video_output_folder, f"frame_{frame_count:04d}")
    if not os.path.exists(frame_output_folder):
        os.makedirs(frame_output_folder)

    # Simpan frame yang diproses ke folder frame
    frame_filename_hist_eq = os.path.join(frame_output_folder, f"frame_{frame_count:04d}_hist_eq.png")
    frame_filename_contrast_stretch = os.path.join(frame_output_folder, f"frame_{frame_count:04d}_contrast_stretch.png")
    frame_filename_clahe = os.path.join(frame_output_folder, f"frame_{frame_count:04d}_clahe.png")

    # Simpan hasil filter
    cv2.imwrite(frame_filename_hist_eq, filtered_hist_eq)
    cv2.imwrite(frame_filename_contrast_stretch, filtered_contrast_stretch)
    cv2.imwrite(frame_filename_clahe, filtered_clahe)

# Fungsi untuk memproses video
def process_video(input_folder_path, filename, output_folder_path):
    video_path = os.path.join(input_folder_path, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Tidak bisa membuka video {filename}")
        return

    # List untuk menyimpan MSE dan PSNR tiap frame
    mse_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}
    psnr_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}

    # Buat folder khusus untuk setiap video
    video_output_folder = os.path.join(output_folder_path, filename.split('.')[0])  # Nama folder berdasarkan nama video
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    frame_count = 0  # Untuk melacak nomor frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Proses frame
        process_frame(frame, frame_count, video_output_folder, mse_values, psnr_values)

        frame_count += 1

    cap.release()

    # Menghitung rata-rata MSE dan PSNR dari semua frame
    avg_mse_hist_eq = np.mean(mse_values['hist_eq'])
    avg_mse_contrast_stretch = np.mean(mse_values['contrast_stretch'])
    avg_mse_clahe = np.mean(mse_values['clahe'])

    avg_psnr_hist_eq = np.mean(psnr_values['hist_eq'])
    avg_psnr_contrast_stretch = np.mean(psnr_values['contrast_stretch'])
    avg_psnr_clahe = np.mean(psnr_values['clahe'])

    # Menyimpan hasil rata-rata MSE dan PSNR ke file .txt
    result_filename = os.path.join(video_output_folder, f"{filename.split('.')[0]}_metrics.txt")
    with open(result_filename, 'w') as result_file:
        result_file.write(f"Rata-rata MSE dan PSNR untuk video: {filename}\n")
        result_file.write(f"Histogram Equalization - MSE: {avg_mse_hist_eq:.2f}, PSNR: {avg_psnr_hist_eq:.2f} dB\n")
        result_file.write(f"Contrast Stretching - MSE: {avg_mse_contrast_stretch:.2f}, PSNR: {avg_psnr_contrast_stretch:.2f} dB\n")
        result_file.write(f"CLAHE - MSE: {avg_mse_clahe:.2f}, PSNR: {avg_psnr_clahe:.2f} dB\n")

    print(f"Metrics saved for video: {filename}")


# Input dan Output folder di Google Drive

input_folder_path = "C:\\Users\\FAFA\\Documents\\Kuliah\\SEMESTER 7\\Comvis\\UTS\\JavelinThrow"
output_folder_path = "C:\\Users\\FAFA\\Documents\\Kuliah\\SEMESTER 7\\Comvis\\UTS\\Output"

# Membuat output folder jika belum ada
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Proses semua video dalam folder input
for filename in os.listdir(input_folder_path):
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        process_video(input_folder_path, filename, output_folder_path)
