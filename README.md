# IMDb Movie Rating Classification

Bu proje, IMDb'den alınan film verileri üzerinde **IMDb puanına göre sınıflandırma** yapmayı amaçlamaktadır. Filmler; yapım yılı, süre, tür ve oy sayısı gibi özellikleri kullanılarak **yüksek puanlı (IMDb ≥ 7)** ve **düşük/orta puanlı (IMDb < 7)** olarak iki sınıfa ayrılmıştır.

## 1. Proje Amacı

- IMDb puanı, oy sayısı, süre ve tür gibi özelliklere bakarak filmlerin:
  - Yüksek puanlı olup olmadığını tahmin etmek (binary classification),
  - Veri setini keşfedici olarak analiz etmek (EDA),
  - Temel istatistiksel analizleri gerçekleştirmek.
- Kullanılan yöntemler:
  - Keşifsel Veri Analizi (EDA)
  - Logistic Regression
  - Random Forest Classifier

## 2. Veri Seti

Kullanılan veri seti `imdb_clean_2000.csv` dosyasından oluşmaktadır ve 2000 film içermektedir.

Ana değişkenler:
- `primaryTitle`: Filmin adı
- `startYear`: Yapım yılı
- `runtimeMinutes`: Süre (dakika)
- `genres`: Tür(ler) (virgülle ayrılmış)
- `averageRating`: IMDb puanı
- `numVotes`: IMDb oy sayısı

Önişleme aşamasında eklenen türetilmiş değişkenler:
- `high_rating`: IMDb puanına göre sınıf etiketi (1: IMDb ≥ 7, 0: IMDb < 7)
- `numVotes_log`: `numVotes` değişkeninin log dönüşümlü hali (`log1p`)
- `primary_genre`: `genres` sütunundan elde edilen ana tür (ilk tür)

Veri setinde eksik değer bulunmamaktadır.

## 3. Veri Önişleme Adımları

1. **Eksik verilerin kontrolü**  
   - `df.info()` ile tüm sütunlar kontrol edildi; veri setinde eksik değer olmadığı görüldü.

2. **Hedef değişkenin oluşturulması**  
   - `averageRating` sürekli değişkeninden ikili hedef üretildi:
     - `high_rating = 1` → `averageRating ≥ 7.0`
     - `high_rating = 0` → `averageRating < 7.0`

3. **Aykırı değer ve süre analizi**  
   - `runtimeMinutes` için özet istatistikler incelendi.
   - Sürelerin çoğu 80–130 dakika aralığında, birkaç uzun metrajlı film outlier konumunda.

4. **Oy sayısı için log dönüşümü**  
   - `numVotes` çok sağa çarpık dağıldığı için:
     - `numVotes_log = log(1 + numVotes)` şeklinde yeni değişken üretildi.
   - Bu sayede oy sayısı daha dengeli bir ölçekte modele sokuldu.

5. **Tür bilgisinin basitleştirilmesi**  
   - `genres` sütunundan ana tür çekildi:
     - `primary_genre = genres.split(",")[0]`
   - Bu değişken, ileride One-Hot Encoding ile sayısallaştırılacak.

## 4. Veri Görselleştirme (EDA)

Proje kapsamında aşağıdaki temel grafikler oluşturulmuştur:

- **IMDb puanı dağılımı (histogram)**  
  - Puanların çoğu 6–8 aralığında yoğunlaşmaktadır.

- **Yüksek / düşük puanlı film dağılımı (bar chart)**  
  - Sınıf dağılımı yaklaşık:
    - `high_rating = 0`: %55
    - `high_rating = 1`: %45  
  - Belirgin bir sınıf dengesizliği yoktur.

- **Film süreleri (histogram + boxplot)**  
  - Çoğu film 80–130 dakika aralığında.
  - Birkaç aşırı uzun film outlier olarak gözlenmektedir.

- **Oy sayısı dağılımı (ham ve log dönüşümlü)**  
  - Ham `numVotes` çok çarpık; az sayıda film çok yüksek oy almış.
  - `numVotes_log` ile dağılım daha dengeli hale gelmiştir.

- **Türlere göre ortalama IMDb puanı (bar chart)**  
  - Documentary, Drama, Thriller gibi türler ortalamada daha yüksek IMDb puanına sahiptir.
  - Bazı türlerin ortalama puanları görece daha düşüktür (ör. Horror, Music).

- **IMDb puanı vs log(oy sayısı) scatter plot**  
  - Daha fazla oy alan filmler genelde 6.5–8.0 puan bandında toplanmaktadır.
  - Az oy alan filmlerin puanları daha dağınıktır.

## 5. Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- (İlerleyen adımlarda) scikit-learn:
  - Logistic Regression
  - Random Forest
  - Train/Test Split
  - Pipeline & ColumnTransformer

## 6. Proje Yapısı (Önerilen)

```text
.
├─ data/
│  └─ imdb_clean_2000.csv
├─ notebooks/
│  ├─ 01_preprocessing_and_eda.ipynb
│  └─ 02_modeling.ipynb
├─ src/
│  └─ model_pipeline.py
└─ README.md
