import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np

# Klasör ve dosya yollarını tanımla
original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"

# Orjinal ve tespit edilen görüntü isimlerini al
original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)

# Çizgi kalınlığı ve grafik stili ayarları
line_thickness = 2
plt.style.use("ggplot")

# Hough dönüşümü için gerekli değişkenlerin tanımlanması
thetas = np.deg2rad(np.arange(-90.0, 90.0))  # Theta aralığı
cos_theta = np.cos(thetas)
sin_theta = np.sin(thetas)
num_thetas = len(thetas)
max_n = 100
sigma = 0.3

# Barkod çizgilerini bulan ana fonksiyon
def detect_barcode_lines():
    for image_name in original_subset_image_names:
        print(image_name)

        # Orjinal ve tespit edilen görüntüleri oku
        original_img = cv2.imread(original_subset_dir + '/' + image_name)
        detected_img = cv2.imread(detection_subset_dir + '/' + image_name)

        # Giriş görüntüsünden kenar haritasını elde et
        edges, plot_input = obtain_edge_map(original_img)

        # Kenar haritasını tespit edilen görüntüyle maskele
        masked_img = cv2.bitwise_and(detected_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # Hough dönüşümünü kenar haritası üzerinde uygula
        accumulator, rhos = find_lines(cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY))

        # Hough uzayını görüntü uzayına dönüştür
        plot_input = hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input)

        # Kenar haritasında konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # En büyük konturu bul
        largest_contour = max(contours, key=cv2.contourArea)

        # Sonucu göster
        plt.imshow(cv2.cvtColor(plot_input, cv2.COLOR_BGR2RGB))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()

        # Sonucu dosyaya kaydet
        cv2.imwrite('barcode_detect/{}.png'.format(image_name), plot_input)

        # Pencereyi kapat
        plt.close()


# Kenar haritasını elde etme fonksiyonu
def obtain_edge_map(img):
    # Gürültüyü azaltmak için görüntüyü bulanıklaştır
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_img, 7)

    # Canny kenar tespiti uygula
    median = np.median(img)
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Görüntü ve kenar haritasını birleştirerek görselleştirme amaçlı görüntü oluştur
    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    return edges, plot_input


# Kenar haritasında çizgileri bulan fonksiyon
def find_lines(img):
    width = img.shape[0]
    height = img.shape[1]

    # Görüntünün çapraz uzunluğu
    diagonal_length = int(np.ceil(np.sqrt(width * width + height * height)))

    # Rho, görüntünün -diagonal_length ile +diagonal_length aralığında olacak şekilde ayarlanır
    rhos = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2)

    # Hough dönüşümü için oylama biriktirici: 2 * diagonal_length satır, num_thetas sütun
    accumulator = np.zeros((2 * diagonal_length, num_thetas), dtype=np.uint64)

    # Kenar haritasında sıfır olmayan (beyaz) noktaların (satır, sütun) indekslerini al
    y_idx, x_idx = np.nonzero(img)

    # Oylama işlemini gerçekleştir
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        # Mevcut rho değerini hesapla
        curr_rhos = np.add(np.array(cos_theta) * x, np.array(sin_theta) * y)

        for t_idx in range(num_thetas):
            # En yakın rho ve thetaya sahip olanın oylamasını bir artır
            accumulator[int(round(curr_rhos[t_idx]) + diagonal_length), t_idx] += 1
    return accumulator, rhos


# Hough uzayını görüntü uzayına dönüştüren fonksiyon
def hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input):
    # Eşik değerini al
    threshold = get_avg_threshold(accumulator)

    # Eşik değerine göre en net çizgilerin indekslerini al
    y_idx, x_idx = np.where(accumulator >= threshold)

    for i in range(len(y_idx)):
        # Düzleştirilmiş indeksi alarak rho ve theta'yı bul
        flatten_idx = get_flatten_idx(accumulator.shape[1], y_idx[i], x_idx[i])

        # Mevcut çizginin rho ve theta parametrelerini bul
        rho = rhos[int(round(flatten_idx / accumulator.shape[1]))]
        theta = thetas[flatten_idx % accumulator.shape[1]]

        # Hough uzayını görüntü uzayına dönüştür
        cos = math.cos(theta)
        sin = math.sin(theta)
        x = cos * rho
        y = sin * rho
        from_point = (int(x + 1000 * (-sin)), int(y + 1000 * cos))
        to_point = (int(x - 1000 * (-sin)), int(y - 1000 * cos))

        # Orjinal ve tespit edilen görüntüler üzerine çizgileri çiz
        cv2.line(original_img, from_point, to_point, (0, 0, 255), line_thickness)
        cv2.line(detected_img, from_point, to_point, (0, 0, 255), line_thickness)

    # Görselleştirme amaçlı görüntüyü oluştur
    plot_input = np.concatenate((plot_input, original_img), axis=1)
    plot_input = np.concatenate((plot_input, detected_img), axis=1)
    return plot_input


# En büyük n indeksini al
def get_n_max_idx(arr):
    # arr içinde n adet en büyük sayıyı bul
    indices = np.argpartition(arr, arr.size - max_n, axis=None)[-max_n:]
    return np.column_stack(np.unravel_index(indices, arr.shape))


# Ortalama eşiğini hesapla
def get_avg_threshold(arr):
    # arr içindeki en büyük n sayısının ortalamasını al
    n_max_idx = get_n_max_idx(arr)

    # Maksimum n sayılarının ortalamasını hesapla
    sum = 0.0
    for i in range(len(n_max_idx)):
        sum += arr[n_max_idx[i][0]][n_max_idx[i][1]]
    return sum / len(n_max_idx)


# Düzleştirilmiş indeksi al
def get_flatten_idx(total_cols, row, col):
    return (row * total_cols) + col


if __name__ == '__main__':
    detect_barcode_lines()
