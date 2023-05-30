
import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np



# Klasör ve dosya yollarını tanımla
original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"
original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)

# Çizgi kalınlığı ve grafik stili ayarları
line_thickness = 2
plt.style.use("ggplot")

# Hough dönüşümü için gerekli değişkenlerin tanımlanması
thetas = np.deg2rad(np.arange(-90.0, 90.0))   # Theta range
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
        edges, plot_input = obtain_edge_map(detected_img)

        # Kenar haritasını tespit edilen görüntüyle maskele
        masked_img = cv2.bitwise_and(detected_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # Hough dönüşümünü kenar haritası üzerinde uygula
        accumulator, rhos = find_lines(cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY))

        # Hough uzayını görüntü uzayına dönüştür
        plot_input = hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input)

        # Kenar haritasında konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # En uygun sınırlayıcı kutuyu bul
        largest_area = 0
        best_box = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > largest_area:
                largest_area = area
                best_box = (x, y, w, h)
                
        # Orijinal ve algılanan görüntülere sınırlayıcı kutu çiz
        if best_box is not None:
            x, y, w, h = best_box
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)
            cv2.rectangle(detected_img, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)
        
        # Sonucu göster
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()
        
        # Sınırlayıcı kutu koordinatlarını kullanarak dikdörtgen bölgeyi kırpma
        cropped_img = original_img[y:y+h, x:x+w]

        # Kırpılan görüntünün bir kopyasını oluşturma ve dikdörtgen çizme
        result_img = cropped_img.copy()
        cv2.rectangle(result_img, (0, 0), (w, h), (0, 255, 0), line_thickness)
        
        # Kırpılan resmi göster
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Cropped Image'), plt.xticks([]), plt.yticks([])
        plt.show()   
        
        # Kırpılan görüntülerin kaydedileceği dizin
        cropped_images_dir = "cropped_images"

        # Dizin yoksa oluştur
        if not os.path.exists(cropped_images_dir):
            os.makedirs(cropped_images_dir)

        # Kırpılan her görüntü için unik bir dosya adı oluşturun
        cropped_image_filename = os.path.join(cropped_images_dir, image_name)

        # Kırpılan görüntüyü kaydet
        cv2.imwrite(cropped_image_filename, cropped_img)
        
        # Orjinal görüntülerin kaydedileceği dizini tanımla
        original_images_dir = "original_images"

        # Dizin yoksa oluştur
        if not os.path.exists(original_images_dir):
            os.makedirs(original_images_dir)

        # Her görüntü için unik bir dosya adı oluşturun
        original_image_filename = os.path.join(original_images_dir, image_name)

        # Orjinal görüntüyü içine kaydet
        cv2.imwrite(original_image_filename, original_img)



def obtain_edge_map(img):
    # Blur the image to reduce noise
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_img, 7)

    # Apply Canny edge detection to image
    median = np.median(img)
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Concat original and edge image for plot
    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    return edges, plot_input

# Kenar haritasını elde etme
def find_lines(img):
    width = img.shape[0]
    height = img.shape[1]

    # Görüntünün köşegen uzunluğu
    diagonal_length = int(np.ceil(np.sqrt(width * width + height * height)))

    # Rho aralığını görüntünün -diagonal_length ile +diagonal uzunluğu arasında değişir
    rhos = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2)

    # Hough dönüşümü oy toplayıcı: 2 * diagonal_length rows, num_thetas columns
    accumulator = np.zeros((2 * diagonal_length, num_thetas), dtype=np.uint64)

    # Sıfır olmayan kenar endekslerine (satır, sütun) dizinleri alın
    y_idx, x_idx = np.nonzero(img)

    # Akümülatörde oy verme
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        # Mevcut rhos'u hesapla
        curr_rhos = np.add(np.array(cos_theta) * x, np.array(sin_theta) * y)

        for t_idx in range(num_thetas):
            # Theta nın ve en yakın rho'nun oyu birer birer artırılır
            accumulator[int(round(curr_rhos[t_idx]) + diagonal_length), t_idx] += 1
    return accumulator, rhos


def hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input):
    #  threshold hesapla
    threshold = get_avg_threshold(accumulator)

    # Threshold a göre en net çizgilerin indekslerini al
    y_idx, x_idx = np.where(accumulator >= threshold)

    for i in range(len(y_idx)):
        # Rho ve theta'yı bulmak için düzleştirme indeksini alın
        flatten_idx = get_flatten_idx(accumulator.shape[1], y_idx[i], x_idx[i])

        # Geçerli satırın rho ve theta parametresini bulun
        rho = rhos[int(round(flatten_idx / accumulator.shape[1]))]
        theta = thetas[flatten_idx % accumulator.shape[1]]

        # Hough uzayını görüntü uzayına çevir
        cos = math.cos(theta)
        sin = math.sin(theta)
        x = cos * rho
        y = sin * rho
        from_point = (int(x + 1000 * (-sin)), int(y + 1000 * cos))
        to_point = (int(x - 1000 * (-sin)), int(y - 1000 * cos))
        
        """
        # Orijinal ve algılanan görüntülerin üzerine çizgiler çiz
        cv2.line(original_img, from_point, to_point, (0, 0, 255), line_thickness)
        """
        cv2.line(detected_img, from_point, to_point, (0, 0, 255), line_thickness)
             
    # Görselleştirme amaçlı görüntüyü oluştur
    plot_input = np.concatenate((plot_input, original_img), axis=1)
    plot_input = np.concatenate((plot_input, detected_img), axis=1)
    return plot_input

# En büyük n indeksini al
def get_n_max_idx(arr):
    # Arr cinsinden n tane maksimum sayıyı bulun
    indices = np.argpartition(arr, arr.size - max_n, axis=None)[-max_n:]
    return np.column_stack(np.unravel_index(indices, arr.shape))

# Ortalama eşiğini hesapla
def get_avg_threshold(arr):
    # Arr'daki maksimum sayıların max_n sayısını alın
    n_max_idx = get_n_max_idx(arr)

    # Max_n sayıların ortalamasını hesapla
    sum = 0.0
    for i in range(len(n_max_idx)):
        sum += arr[n_max_idx[i][0]][n_max_idx[i][1]]
    return sum/len(n_max_idx)

# Satır ve sütun indekslerini tek bir indeks olarak döndür
def get_flatten_idx(total_cols, row, col):
    return (row * total_cols) + col

# Ana işlevi çağırıyoruz.
if __name__ == '__main__':
    detect_barcode_lines()
    

