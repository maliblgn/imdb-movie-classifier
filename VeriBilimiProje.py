import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 0) VERİYİ YÜKLE VE TEMEL ÖNİŞLEME
#    - high_rating (0/1) hedef değişkenini oluştur
#    - runtimeMinutes = 0 olanları temizle (gerekirse)
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

# Süresi 0 olan filmleri temizle (mantıksız kayıtlar olsaydı)
df = df[df["runtimeMinutes"] > 0].copy()

# Oy sayısına log dönüşümü
df["numVotes_log"] = np.log1p(df["numVotes"])

# Tür bilgisinden ana türü çıkar
df["primary_genre"] = df["genres"].str.split(",").str[0]

# Analizde kullanacağımız sayısal sütunlar
num_cols = ["averageRating", "runtimeMinutes", "numVotes_log", "startYear"]

# Matplotlib genel ayarları (isteğe bağlı)
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True


# ============================================================
# 1) VERİ GÖRSELLEŞTİRME (EDA)
#    Amaç: Dağılımları ve temel ilişkileri görsel olarak incelemek
# ============================================================

# Grafik 1: IMDb puanlarının genel dağılımı (averageRating histogram)
# Amaç: Filmler hangi puan aralığında yoğunlaşıyor?
plt.figure()
plt.hist(df["averageRating"], bins=20)
plt.xlabel("IMDb Puanı (averageRating)")
plt.ylabel("Film Sayısı")
plt.title("IMDb Puanı Dağılımı")
plt.show()


# Grafik 2: Yüksek/düşük puanlı film sınıflarının dağılımı (high_rating bar chart)
# Amaç: Sınıflar dengeli mi, dengesiz mi?
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


# Grafik 3: Film sürelerinin dağılımı (runtimeMinutes histogram)
# Amaç: Filmlerin çoğu hangi süre aralığında?
plt.figure()
plt.hist(df["runtimeMinutes"], bins=20)
plt.xlabel("Süre (dakika)")
plt.ylabel("Film Sayısı")
plt.title("Film Süresi Dağılımı")
plt.show()


# Grafik 4: Film sürelerindeki olası uç değerler (runtimeMinutes boxplot)
# Amaç: Aşırı uzun/kısa filmleri görselleştirmek.
plt.figure()
plt.boxplot(df["runtimeMinutes"], vert=False)
plt.xlabel("Süre (dakika)")
plt.title("Film Süresi Boxplot")
plt.show()


# Grafik 5: Oy sayılarının ham dağılımı (numVotes histogram)
# Amaç: Oy sayıları çarpık mı, geniş mi dağılıyor?
plt.figure()
plt.hist(df["numVotes"], bins=20)
plt.xlabel("Oy Sayısı (numVotes)")
plt.ylabel("Film Sayısı")
plt.title("Oy Sayısı Dağılımı (Ham)")
plt.show()


# Grafik 6: Log dönüşüm uygulanmış oy sayısı dağılımı (numVotes_log histogram)
# Amaç: Log dönüşüm sonrası dağılım daha dengeli mi?
plt.figure()
plt.hist(df["numVotes_log"], bins=20)
plt.xlabel("Log(1 + Oy Sayısı) (numVotes_log)")
plt.ylabel("Film Sayısı")
plt.title("Oy Sayısı Dağılımı (Log Dönüşüm Sonrası)")
plt.show()


# Grafik 7: Türlere göre ortalama IMDb puanı (primary_genre bar chart)
# Amaç: Hangi türler ortalamada daha yüksek puanlı?
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


# Grafik 8: IMDb puanı ile log(oy sayısı) ilişkisi (scatter plot)
# Amaç: Daha çok oy alan filmler belirli bir puan aralığında mı toplanıyor?
plt.figure()
plt.scatter(df["numVotes_log"], df["averageRating"], alpha=0.5)
plt.xlabel("Log(1 + Oy Sayısı) (numVotes_log)")
plt.ylabel("IMDb Puanı (averageRating)")
plt.title("IMDb Puanı vs Oy Sayısı (Log Ölçek)")
plt.show()


# ============================================================
# 2) VERİNİN İSTATİSTİKSEL ANALİZLERİ
#    Amaç: Tanımlayıcı istatistikler, normallik ve korelasyon
# ============================================================

# 2.1) Merkezi eğilim ve temel tanımlayıcı istatistikler
#      (ortalama, medyan, min, max, std, çeyrekler)
desc = df[num_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
print("\nSayısal değişkenler için tanımlayıcı istatistikler:")
print(desc)

# Mod (mode) örneği: IMDb puanları için
rating_mode = df["averageRating"].mode()
print("\nIMDb puanlarının modu (en çok tekrar eden değerler):")
print(rating_mode.values)

# Sınıfa göre (high_rating) ortalama süre ve puan
group_summary = df.groupby("high_rating")[["averageRating", "runtimeMinutes", "numVotes_log"]].mean()
print("\nSınıflara göre ortalama değerler (0=düşük/orta, 1=yüksek):")
print(group_summary)


# 2.2) Değişkenlik, çarpıklık (skewness) ve basıklık (kurtosis)
#      Amaç: Varyans, std, aralık, çarpıklık, basıklık hesaplamak.
variance = df[num_cols].var()
std_dev  = df[num_cols].std()
data_min = df[num_cols].min()
data_max = df[num_cols].max()
value_range = data_max - data_min

skewness = df[num_cols].skew()       # Çarpıklık
kurtosis = df[num_cols].kurtosis()   # Basıklık (excess kurtosis)

stats_table = pd.DataFrame({
    "variance": variance,
    "std_dev": std_dev,
    "min": data_min,
    "max": data_max,
    "range": value_range,
    "skewness": skewness,
    "kurtosis": kurtosis
})

print("\nDeğişkenlik, çarpıklık ve basıklık özet tablosu:")
print(stats_table)


# 2.3) Normallik incelemesi (Shapiro-Wilk testi)
#      H0: Veri normal dağılmıştır.
#      p < 0.05 → H0 reddedilir (normal değil).
print("\nShapiro-Wilk normallik testi sonuçları:")

for col in ["averageRating", "runtimeMinutes", "numVotes_log"]:
    stat, p = stats.shapiro(df[col])
    print(f"\nDeğişken: {col}")
    print("  Test istatistiği:", stat)
    print("  p-değeri:", p)
    if p < 0.05:
        print("  → p < 0.05: Normal dağılım varsayımı reddedilir (normal değil).")
    else:
        print("  → p ≥ 0.05: Normal dağılım varsayımı reddedilemez (normal kabul edilebilir).")


# 2.4) Korelasyon analizi
#      Amaç: IMDb puanı ile diğer sayısal değişkenler arasındaki
#             doğrusal ilişkiyi (korelasyon katsayısı) görmek.
corr_matrix = df[["averageRating", "runtimeMinutes", "numVotes_log", "startYear"]].corr()
print("\nKorelasyon matrisi:")
print(corr_matrix)

print("\nIMDb puanı ile diğer değişkenlerin korelasyonu:")
print(corr_matrix["averageRating"])

# ============================================================
# 3) SINIFLANDIRMA MODELİNİ OLUŞTURMA
#    Amaç: Kullanılacak özellikleri (X) ve hedefi (y) tanımlayıp,
#           Logistic Regression ve Random Forest için
#           ön işleme + model pipeline yapısını kurmak.
#
#    Bu aşamada:
#      - Sadece model "yapısı" oluşturuluyor.
#      - Henüz eğitim (fit) ve test değerlendirmesi yapılmıyor.
#        (Bunlar bir sonraki adımda: Eğitim/Test Aşamaları)
# ============================================================

# Modelde kullanacağımız sayısal ve kategorik özellikler
numeric_features = ["startYear", "runtimeMinutes", "numVotes_log"]
categorical_features = ["primary_genre"]

# Özellik matrisi (X) ve hedef vektör (y)
X = df[numeric_features + categorical_features]
y = df["high_rating"]

print("\nÖzellik matrisi (X) ve hedef (y) hazırlandı.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# Ön işleme adımları:
# - Sayısal özellikler: StandardScaler ile ölçekleme
# - Kategorik özellikler: OneHotEncoder ile one-hot kodlama
# -----------------------------

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# MODEL 1: Logistic Regression
# -----------------------------
log_reg_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),            # Önce ön işleme
        ("clf", LogisticRegression(max_iter=1000))
    ]
)

# -----------------------------
# MODEL 2: Random Forest Classifier
# -----------------------------
rf_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]
)

print("\nSınıflandırma modellerinin pipeline yapıları oluşturuldu.")
print("\nLogistic Regression Pipeline:")
print(log_reg_model)

print("\nRandom Forest Pipeline:")
print(rf_model)

# ============================================================
# 4) SINIFLANDIRMA MODELİNİN EĞİTİMİ VE TEST EDİLMESİ
#    Amaç:
#      - Veriyi eğitim ve test olarak ayırmak
#      - Logistic Regression ve Random Forest modellerini eğitmek
#      - Test seti üzerinde temel sınıflandırma metriklerini
#        (accuracy, precision, recall, F1) hesaplamak
# ============================================================

# 4.1) Eğitim / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # %80 eğitim / %20 test
    stratify=y,         # sınıf dengesini korumak için
    random_state=42
)

print("\nEğitim/Test ayrımı tamamlandı.")
print("Eğitim seti boyutu:", X_train.shape[0])
print("Test seti boyutu   :", X_test.shape[0])

# 4.2) Logistic Regression modelinin eğitimi ve değerlendirilmesi
log_reg_model.fit(X_train, y_train)
y_pred_log = log_reg_model.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print("\n=== LOGISTIC REGRESSION – Test Sonuçları ===")
print(f"Accuracy : {acc_log:.4f}")
print(f"Precision: {prec_log:.4f}")
print(f"Recall   : {rec_log:.4f}")
print(f"F1-score : {f1_log:.4f}")

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix (Logistic Regression):")
print(cm_log)

# 4.3) Random Forest modelinin eğitimi ve değerlendirilmesi
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n=== RANDOM FOREST – Test Sonuçları ===")
print(f"Accuracy : {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall   : {rec_rf:.4f}")
print(f"F1-score : {f1_rf:.4f}")

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (Random Forest):")
print(cm_rf)

# 4.4) Modellerin özet karşılaştırması (test seti)
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_log, acc_rf],
    "Precision": [prec_log, prec_rf],
    "Recall": [rec_log, rec_rf],
    "F1-score": [f1_log, f1_rf]
})

print("\n=== MODEL KARŞILAŞTIRMA (Test Seti) ===")
print(comparison_df.to_string(index=False))
