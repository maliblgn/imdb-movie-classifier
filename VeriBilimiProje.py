import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 0) VERİYİ YÜKLE VE ADIM 1'DEKİ TEMEL ÖNİŞLEMEYİ TEKRARLA
#    - high_rating (0/1) hedef değişkenini oluştur
#    - runtimeMinutes = 0 olanları temizle
#    - numVotes_log (log dönüşüm) oluştur
#    - primary_genre (ana tür) çıkar
# ============================================================

# Script dosyasının dizinini al ve CSV dosyasını bu dizinden oku
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "imdb_clean_2000.csv")
df = pd.read_csv(csv_path)

# Hedef değişken: IMDb puanı 7 ve üzeri olanlara 1, diğerlerine 0
threshold = 7.0
df["high_rating"] = (df["averageRating"] >= threshold).astype(int)

# Süresi 0 olan filmleri temizle (mantıksız kayıtlar)
df = df[df["runtimeMinutes"] > 0].copy()

# Oy sayısına log dönüşümü
df["numVotes_log"] = np.log1p(df["numVotes"])

# Tür bilgisinden ana türü çıkar
df["primary_genre"] = df["genres"].str.split(",").str[0]

# (İsteğe bağlı) Grafik ayarları
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True


# ============================================================
# Grafik 1: IMDb puanlarının genel dağılımını görmek için
#           averageRating sütunundan histogram çiziyoruz.
#           Amaç: Filmler hangi puan aralığında yoğunlaşıyor?
# ============================================================

plt.figure()
plt.hist(df["averageRating"], bins=20)
plt.xlabel("IMDb Puanı (averageRating)")
plt.ylabel("Film Sayısı")
plt.title("IMDb Puanı Dağılımı")
plt.show()


# ============================================================
# Grafik 2: Yüksek/düşük puanlı film sınıflarının dağılımını
#           görmek için high_rating (0/1) üzerinden bar grafiği.
#           Amaç: Sınıflar dengeli mi, dengesiz mi?
# ============================================================

class_counts = df["high_rating"].value_counts().sort_index()

plt.figure()
plt.bar(["Düşük/Orta (<7)", "Yüksek (≥7)"], class_counts.values)
plt.xlabel("Sınıf")
plt.ylabel("Film Sayısı")
plt.title("Yüksek/Düşük Puanlı Film Dağılımı")
plt.show()

print("Sınıf sayıları:")
print(class_counts)
print("\nSınıf oranları:")
print(df["high_rating"].value_counts(normalize=True))


# ============================================================
# Grafik 3: Film sürelerinin dağılımını görmek için
#           runtimeMinutes üzerinden histogram.
#           Amaç: Filmlerin çoğu hangi süre aralığında?
# ============================================================

plt.figure()
plt.hist(df["runtimeMinutes"], bins=20)
plt.xlabel("Süre (dakika)")
plt.ylabel("Film Sayısı")
plt.title("Film Süresi Dağılımı")
plt.show()


# ============================================================
# Grafik 4: Film sürelerindeki olası uç değerleri (outlier)
#           görmek için runtimeMinutes üzerinden boxplot.
#           Amaç: Aşırı uzun/kısa filmleri görselleştirmek.
# ============================================================

plt.figure()
plt.boxplot(df["runtimeMinutes"], vert=False)
plt.xlabel("Süre (dakika)")
plt.title("Film Süresi Boxplot")
plt.show()


# ============================================================
# Grafik 5: Oy sayılarının ham dağılımını görmek için
#           numVotes üzerinden histogram.
#           Amaç: Oy sayıları çarpık mı, geniş mi dağılıyor?
# ============================================================

plt.figure()
plt.hist(df["numVotes"], bins=20)
plt.xlabel("Oy Sayısı (numVotes)")
plt.ylabel("Film Sayısı")
plt.title("Oy Sayısı Dağılımı (Ham)")
plt.show()


# ============================================================
# Grafik 6: Log dönüşüm uygulanmış oy sayılarının dağılımını
#           görmek için numVotes_log üzerinden histogram.
#           Amaç: Log dönüşüm sonrası dağılım daha dengeli mi?
# ============================================================

plt.figure()
plt.hist(df["numVotes_log"], bins=20)
plt.xlabel("Log(1 + Oy Sayısı) (numVotes_log)")
plt.ylabel("Film Sayısı")
plt.title("Oy Sayısı Dağılımı (Log Dönüşüm Sonrası)")
plt.show()


# ============================================================
# Grafik 7: Türlere göre ortalama IMDb puanlarını göstermek için
#           primary_genre bazında averageRating ortalamasını alıp
#           en yüksek ortalamaya sahip ilk 10 tür için bar grafiği.
#           Amaç: Hangi türler ortalamada daha yüksek puanlı?
# ============================================================

genre_mean_rating = df.groupby("primary_genre")["averageRating"].mean().sort_values(ascending=False)

top_n = 10
genre_mean_top = genre_mean_rating.head(top_n)

plt.figure()
plt.bar(genre_mean_top.index, genre_mean_top.values)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Ana Tür (primary_genre)")
plt.ylabel("Ortalama IMDb Puanı")
plt.title(f"Türlere Göre Ortalama IMDb Puanı (İlk {top_n})")
plt.tight_layout()
plt.show()

print("\nTürlere göre ortalama IMDb puanı (ilk 10):")
print(genre_mean_top)


# ============================================================
# Grafik 8 (İsteğe Bağlı): IMDb puanı ile log oy sayısı arasındaki
#           ilişkiyi görmek için numVotes_log vs averageRating
#           scatter plot.
#           Amaç: Daha çok oy alan filmler, belirli bir puan
#                  aralığında mı toplanıyor, dağınık mı?
# ============================================================

plt.figure()
plt.scatter(df["numVotes_log"], df["averageRating"], alpha=0.5)
plt.xlabel("Log(1 + Oy Sayısı) (numVotes_log)")
plt.ylabel("IMDb Puanı (averageRating)")
plt.title("IMDb Puanı vs Oy Sayısı (Log Ölçek)")
plt.show()
