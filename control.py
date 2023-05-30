import os

folder_path = "correct_decode"  # "correct_decode" klasörünüzün yolu
image_extensions = (".jpg", ".jpeg", ".png")  # Görsel dosya uzantıları

image_count = 0

for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_count += 1

print("\nKodu çözülmüş barkod sayısı:", image_count)



folder_path = "wrong_decode"  # "correct_decode" klasörünüzün yolu
image_extensions = (".jpg", ".jpeg", ".png")  # Görsel dosya uzantıları

image_count = 0

for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_count += 1

print("\nKodu çözülmemiş barkod sayısı:", image_count)