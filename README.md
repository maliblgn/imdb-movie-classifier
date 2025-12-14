# IMDb Movie Rating Classification

Bu proje, IMDb'den alınan film verileri üzerinde **IMDb puanına göre sınıflandırma** yapmayı amaçlamaktadır. Filmler; yapım yılı, süre, tür ve oy sayısı gibi özellikleri kullanılarak **yüksek puanlı (IMDb ≥ 7)** ve **düşük/orta puanlı (IMDb < 7)** olarak iki sınıfa ayrılmıştır.

---

## 1. Proje Amacı

- IMDb puanı, oy sayısı, süre ve tür gibi özelliklere bakarak filmlerin:
  - Yüksek puanlı olup olmadığını tahmin etmek (binary classification),
  - Veri setini keşfedici olarak analiz etmek (EDA),
  - Temel istatistiksel analizleri gerçekleştirmek.
- Kullanılan yöntemler:
  - Keşifsel Veri Analizi (EDA)
  - Logistic Regression
  - Random Forest Classifier

---

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

---

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

---

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

---

## 5. Verinin İstatistiksel Analizleri

Bu bölümde, önişlemesi yapılmış IMDb film veri seti üzerinde temel istatistiksel analizler gerçekleştirilmiştir. Amaç, yalnızca görsel inceleme ile yetinmeyip, verinin merkezi eğilimi, değişkenliği, dağılım yapısı ve değişkenler arası ilişkileri istatistiksel olarak da anlamaktır.

### 5.1. Merkezi Eğilim ve Tanımlayıcı İstatistikler

Temel sayısal değişkenler için (örneğin `averageRating`, `runtimeMinutes`, `numVotes_log`, `startYear`) tanımlayıcı istatistikler hesaplanmıştır:

- Ortalama (mean)
- Medyan
- Minimum ve maksimum değerler
- Standart sapma
- Çeyreklikler (25%, 50%, 75%)

Ayrıca IMDb puanlarının modu (en sık görülen değerler) incelenmiş ve `high_rating` sınıfına göre grup bazlı özetler çıkarılmıştır. Örneğin:

- `high_rating = 0` (düşük/orta puanlı filmler) için ortalama puan, süre ve log(oy sayısı)
- `high_rating = 1` (yüksek puanlı filmler) için ortalama puan, süre ve log(oy sayısı)

Bu sayede veri setindeki “tipik” film profili ve yüksek/düşük puanlı filmler arasındaki temel farklar merkezi eğilim açısından özetlenmiştir.

### 5.2. Değişkenlik, Çarpıklık ve Basıklık

Değişkenliğin boyutunu ve dağılımın şeklini anlamak için aynı sayısal değişkenler üzerinde şu ölçüler hesaplanmıştır:

- Varyans
- Standart sapma
- Aralık (range = max – min)
- Çarpıklık (skewness)
- Basıklık (kurtosis)

Çarpıklık değeri, dağılımın simetrik olup olmadığını; basıklık değeri ise dağılımın normal dağılıma göre daha sivri mi yoksa daha basık mı olduğunu göstermektedir. Örneğin:

- `numVotes_log` için pozitif çarpıklık, oy sayısının hâlâ tam simetrik olmadığını ancak ham `numVotes` değişkenine göre çok daha dengeli bir yapı sergilediğini göstermektedir.
- `averageRating` için çarpıklık ve basıklık değerleri, puanların orta–yüksek aralıkta yoğunlaştığını ve çok uç değerlerin sınırlı olduğunu işaret etmektedir.

Bu ölçüler, veri setinin varyasyon düzeyini ve dağılım karakteristiğini nicel olarak ortaya koymaktadır.

### 5.3. Normallik Analizi (Shapiro-Wilk Testi)

Seçilen sayısal değişkenler (`averageRating`, `runtimeMinutes`, `numVotes_log`) için, normal dağılıma uyup uymadıklarını test etmek amacıyla Shapiro-Wilk normallik testi uygulanmıştır.

- Null hipotez (H0): İlgili değişken normal dağılmıştır.
- Alternatif hipotez (H1): İlgili değişken normal dağılmamaktadır.

Her değişken için test istatistiği ve p-değeri hesaplanmış, p-değeri 0.05 eşiği ile karşılaştırılmıştır. Elde edilen sonuçlara göre:

- P-değeri 0.05’in altında çıkan değişkenler için normal dağılım varsayımı reddedilmiş,
- P-değeri 0.05’in üzerinde olan değişkenler için ise “normal dağılıma yakın” davranış sergilediği kabul edilmiştir.

Bu analiz, özellikle daha sonra kullanılabilecek parametrik yöntemler (örneğin bazı regresyon veya istatistiksel testler) açısından veri setinin uygunluğunu değerlendirmek için önemlidir.

### 5.4. Korelasyon Analizi

Son olarak, IMDb puanı ile diğer sayısal özellikler arasındaki doğrusal ilişkiler incelenmiştir. Bu amaçla `averageRating` ile:

- `numVotes_log` (log dönüşümlü oy sayısı),
- `runtimeMinutes` (film süresi),
- `startYear` (yapım yılı)

arasındaki korelasyon katsayıları hesaplanmış ve bir korelasyon matrisi oluşturulmuştur.

Genel yorum çerçevesi:

- |corr| ≈ 0.0–0.3  → zayıf ilişki  
- |corr| ≈ 0.3–0.7  → orta düzey ilişki  
- |corr| > 0.7      → güçlü ilişki  

Bu analiz sonucunda, örneğin:

- `averageRating` ile `numVotes_log` arasında pozitif ve belirli bir düzeyde ilişki bulunması, daha fazla oy alan filmlerin genellikle daha güvenilir ve hafifçe daha yüksek puanlı olduğunu düşündürmektedir.
- `averageRating` ile `runtimeMinutes` veya `startYear` arasındaki korelasyonların görece daha düşük çıkması, bu değişkenlerin IMDb puanı üzerinde sınırlı bir etkisi olabileceğini göstermektedir.

Bu bölüm, verinin istatistiksel açıdan yapısını ortaya koyarak, sonraki adımlarda kurulacak sınıflandırma modelleri için güçlü bir teorik zemin hazırlamaktadır.

---

## 6. Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- (İlerleyen adımlarda) scikit-learn:
  - Logistic Regression
  - Random Forest
  - Train/Test Split
  - Pipeline & ColumnTransformer

---

## 7. Proje Yapısı (Önerilen)

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
