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
    cv2.imwrite(os.path.join(frame_output_folder, f"frame_{frame_count:04d}_hist_eq.png"), filtered_hist_eq)
    cv2.imwrite(os.path.join(frame_output_folder, f"frame_{frame_count:04d}_contrast_stretch.png"), filtered_contrast_stretch)
    cv2.imwrite(os.path.join(frame_output_folder, f"frame_{frame_count:04d}_clahe.png"), filtered_clahe)

# Fungsi untuk memproses video
def process_video(input_folder_path, filename, output_folder_path):
    video_path = os.path.join(input_folder_path, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Tidak bisa membuka video {filename}")
        return None, None

    # List untuk menyimpan MSE dan PSNR tiap frame
    mse_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}
    psnr_values = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}

    # Buat folder khusus untuk setiap video
    video_output_folder = os.path.join(output_folder_path, filename.split('.')[0])
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

    # Mengembalikan mse_values dan psnr_values untuk digunakan di luar fungsi
    return mse_values, psnr_values

# Fungsi utama untuk memproses semua video dalam folder
def main(input_folder_path, output_folder_path):
    # Membuat output folder jika belum ada
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Variabel untuk menyimpan total MSE dan PSNR dari semua video
    overall_mse = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}
    overall_psnr = {'hist_eq': [], 'contrast_stretch': [], 'clahe': []}

    # Proses semua video dalam folder input
    for filename in os.listdir(input_folder_path):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            mse_values, psnr_values = process_video(input_folder_path, filename, output_folder_path)

            if mse_values is not None and psnr_values is not None:
                # Tambahkan hasil tiap video ke hasil keseluruhan
                overall_mse['hist_eq'].extend(mse_values['hist_eq'])
                overall_mse['contrast_stretch'].extend(mse_values['contrast_stretch'])
                overall_mse['clahe'].extend(mse_values['clahe'])

                overall_psnr['hist_eq'].extend(psnr_values['hist_eq'])
                overall_psnr['contrast_stretch'].extend(psnr_values['contrast_stretch'])
                overall_psnr['clahe'].extend(psnr_values['clahe'])

    # Menghitung rata-rata MSE dan PSNR dari semua video
    final_avg_mse_hist_eq = np.mean(overall_mse['hist_eq']) if overall_mse['hist_eq'] else 0
    final_avg_mse_contrast_stretch = np.mean(overall_mse['contrast_stretch']) if overall_mse['contrast_stretch'] else 0
    final_avg_mse_clahe = np.mean(overall_mse['clahe']) if overall_mse['clahe'] else 0

    final_avg_psnr_hist_eq = np.mean(overall_psnr['hist_eq']) if overall_psnr['hist_eq'] else 0
    final_avg_psnr_contrast_stretch = np.mean(overall_psnr['contrast_stretch']) if overall_psnr['contrast_stretch'] else 0
    final_avg_psnr_clahe = np.mean(overall_psnr['clahe']) if overall_psnr['clahe'] else 0

    # Menyimpan hasil rata-rata akhir dari semua video ke file .txt
    final_result_filename = os.path.join(output_folder_path, "overall_metrics.txt")
    with open(final_result_filename, 'w') as final_result_file:
        final_result_file.write("Rata-rata MSE dan PSNR dari semua video:\n")
        final_result_file.write(f"Histogram Equalization - MSE: {final_avg_mse_hist_eq:.2f}, PSNR: {final_avg_psnr_hist_eq:.2f} dB\n")
        final_result_file.write(f"Contrast Stretching - MSE: {final_avg_mse_contrast_stretch:.2f}, PSNR: {final_avg_psnr_contrast_stretch:.2f} dB\n")
        final_result_file.write(f"CLAHE - MSE: {final_avg_mse_clahe:.2f}, PSNR: {final_avg_psnr_clahe:.2f} dB\n")

    print("Overall metrics saved for all videos.")

# Menjalankan fungsi utama
input_folder_path = 'D:\\7_smt.7\\ComVis\\UTS\\diving'
output_folder_path = 'D:\\7_smt.7\\ComVis\\UTS\\divingOut'
main(input_folder_path, output_folder_path)
