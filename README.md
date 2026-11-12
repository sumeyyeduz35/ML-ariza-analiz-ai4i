
# Makine Öğrenmesi ile Makine Arızası Analizi
# Machine Failure Analysis Using Machine Learning

AI4I 2020 veri seti üzerinde makine arızalarının sınıflandırılması, dengesiz veri analizi ve makine öğrenmesi modellerinin karşılaştırılması.

Machine failure classification, imbalanced data analysis, and machine learning model comparison using the AI4I 2020 dataset.

**Teknolojiler / Technologies:** Python | Pandas | NumPy | Scikit-learn | Imbalanced-learn | Matplotlib

---

# Türkçe

## Proje Hakkında

Bu proje, endüstriyel ekipmanların çalışma parametrelerinden yararlanarak makine arızalarının sınıflandırılmasını incelemek amacıyla geliştirilmiştir.

AI4I 2020 Predictive Maintenance veri seti kullanılarak veri ön işleme, sınıf dengesizliği analizi, model eğitimi, hiperparametre optimizasyonu ve performans değerlendirmesi çalışmaları gerçekleştirilmiştir.

Proje kapsamında Support Vector Classifier (SVC) ve Logistic Regression algoritmaları karşılaştırılmıştır.

Çalışma, makine öğrenmesinin kestirimci bakım alanındaki kullanımını ve özellikle nadir görülen arızaların sınıflandırılmasındaki güçlükleri ele almaktadır.

## Veri Seti

Çalışmada AI4I 2020 Predictive Maintenance veri seti kullanılmıştır.

Veri seti 10.000 kayıt içermektedir. Ana analiz için 2.500 ile 7.500 arasındaki kayıtlar seçilmiş ve 5.001 gözlem kullanılmıştır.

### Giriş Değişkenleri

- Hava sıcaklığı (K)
- Proses sıcaklığı (K)
- Dönme hızı (rpm)
- Tork (Nm)
- Takım aşınması (dakika)

### Hedef Değişken

**Machine failure:** Makine arızası durumunu temsil eden ikili sınıflandırma değişkeni.

- 0: Arıza yok
- 1: Arıza var

## Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| Python | Veri analizi ve model geliştirme |
| Pandas | Veri işleme |
| NumPy | Sayısal hesaplamalar |
| Scikit-learn | Model eğitimi ve değerlendirme |
| Imbalanced-learn | SMOTE ile sınıf dengeleme |
| Matplotlib | Veri ve performans görselleştirme |
| Jupyter Notebook | Analiz ve deneylerin yürütülmesi |

## Uygulanan Yöntemler

### Veri Ön İşleme

- Veri setinin yüklenmesi ve yapısının incelenmesi
- Analiz için belirli kayıtların seçilmesi
- Modelleme dışı bırakılan sütunların kaldırılması
- Bağımsız ve bağımlı değişkenlerin ayrılması
- Eğitim ve test verilerinin oluşturulması

Veri seti %80 eğitim ve %20 test olacak şekilde ayrılmıştır.

### Sınıf Dengesizliği

Arıza kayıtlarının normal çalışma kayıtlarına göre daha az olması nedeniyle SMOTE yöntemi kullanılarak eğitim verisindeki azınlık sınıfı için sentetik örnekler oluşturulmuştur.

Bu yaklaşımın arıza tespit performansı üzerindeki etkisi incelenmiştir.

### Makine Öğrenmesi Modelleri

Ana AI4I analizinde iki sınıflandırma algoritması kullanılmıştır:

1. Support Vector Classifier (SVC)
2. Logistic Regression

Modeller eğitim verisi üzerinde eğitilmiş ve ayrılmış test verisi üzerinde değerlendirilmiştir.

### Hiperparametre Optimizasyonu

GridSearchCV kullanılarak SVC modelinin farklı hiperparametre kombinasyonları incelenmiştir.

Değerlendirilen parametreler:

- C
- kernel
- gamma

Ayrıca StandardScaler ile ölçekleme ve permutation importance yöntemiyle özellik önem analizi yapılmıştır.

## Model Performansı

Notebook'ta kayıtlı ilk AI4I test sonuçları aşağıdaki gibidir:

| Model | Accuracy | Arıza Recall | Arıza F1 |
|---|---|---|---|
| SVC + SMOTE | %79,62 | %85 | 0,25 |
| Logistic Regression + SMOTE | %83,12 | %78 | 0,27 |
| Optimize edilmiş SVC | %80,62 | %85 | 0,26 |

Bu sonuçlar 1.001 gözlemden oluşan test kümesinde elde edilmiş kayıtlı notebook çıktılarıdır.

Arıza sınıfının az sayıda örnek içermesi nedeniyle accuracy tek başına yeterli bir performans ölçütü değildir. Özellikle arıza sınıfındaki precision, recall ve F1-score değerleri birlikte değerlendirilmelidir.

## Ek Analizler

Notebook içerisinde aşağıdaki çalışmalar da bulunmaktadır:

- Confusion Matrix analizi
- Model doğruluklarının karşılaştırılması
- F1-score karşılaştırmaları
- Permutation Importance analizi
- Özellik seçimi sonrası yeniden eğitim
- SMOTE öncesi ve sonrası performans karşılaştırmaları
- GridSearchCV ile model optimizasyonu
- Ek sentetik veri ve örnek sınıflandırma veri setleri üzerinde deneyler

Ek veri setleri üzerinde elde edilen sonuçlar, AI4I arıza sınıflandırma sonuçlarından ayrı değerlendirilmelidir.

## Proje Yapısı

```text
ML-ariza-analiz-ai4i/
├── ML-ariza-analiz-ai4i.ipynb
└── ai4i2020.csv
```

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

### Kurulum

Projeyi klonlayın:

```bash
git clone https://github.com/sumeyyeduz35/ML-ariza-analiz-ai4i.git
cd ML-ariza-analiz-ai4i
```

Gerekli Python kütüphanelerini yükleyin:

```bash
python -m pip install notebook pandas numpy scikit-learn imbalanced-learn matplotlib
```

Notebook ortamını başlatın:

```bash
jupyter notebook
```

`ML-ariza-analiz-ai4i.ipynb` dosyasını açarak analizleri inceleyebilirsiniz.

Notebook farklı deneyler içerdiğinden hücreler çalıştırılmadan önce kullanılan veri seti ve değişkenlerin kontrol edilmesi önerilir.

## Teknik Sınırlamalar

Bu çalışma deneysel bir makine öğrenmesi analizidir.

- Ana analizde veri setinin tamamı yerine 5.001 kayıt kullanılmıştır.
- Sınıf dengesizliği model performansını etkilemektedir.
- Bazı özellik seçimi işlemleri test verisine dayanmaktadır.
- SMOTE ve çapraz doğrulama adımlarının veri sızıntısını önleyecek şekilde yeniden düzenlenmesi daha güvenilir performans tahminleri sağlayabilir.
- Notebook'un sonraki bölümlerinde sentetik veriler ve farklı örnek veri setleri de kullanılmıştır.
- Sonuçlar gerçek üretim ortamında doğrulanmış arıza tespit performansı olarak değerlendirilmemelidir.

## Proje Durumu

**Tamamlandı — Deneysel makine öğrenmesi çalışması.**

Proje; veri hazırlama, sınıflandırma, dengesiz veri yönetimi, model karşılaştırma ve değerlendirme yöntemlerini kapsayan bir analiz çalışmasıdır.

---

# English

## About the Project

This project investigates machine failure classification using operational parameters from industrial equipment.

The AI4I 2020 Predictive Maintenance dataset is used to explore data preprocessing, class imbalance, supervised learning, hyperparameter optimization, and model evaluation.

The main analysis compares Support Vector Classifier (SVC) and Logistic Regression models.

The project focuses on practical machine learning techniques for predictive maintenance, with particular attention to the challenges of identifying relatively rare failure events.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance dataset.

The original dataset contains 10,000 records. A subset of 5,001 observations is selected for the primary analysis.

### Input Features

- Air temperature (K)
- Process temperature (K)
- Rotational speed (rpm)
- Torque (Nm)
- Tool wear (minutes)

### Target Variable

**Machine failure:** Binary classification target.

- 0: No failure
- 1: Failure

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and model development |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Model training and evaluation |
| Imbalanced-learn | SMOTE oversampling |
| Matplotlib | Data and performance visualization |
| Jupyter Notebook | Experimentation and analysis |

## Methodology

### Data Preprocessing

The preprocessing workflow includes:

- Loading and examining the dataset
- Selecting a subset of observations
- Removing non-modeling columns
- Defining input features and the target variable
- Splitting data into training and test sets

An 80/20 training-test split is used.

### Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is applied to generate synthetic minority-class examples in the training data.

The impact of class balancing on failure detection is investigated.

### Machine Learning Models

Two primary classification algorithms are evaluated on the AI4I dataset:

1. Support Vector Classifier (SVC)
2. Logistic Regression

Models are trained and evaluated using a held-out test set.

### Hyperparameter Optimization

GridSearchCV is used to investigate different SVC hyperparameter combinations, including:

- C
- kernel
- gamma

The notebook also explores feature scaling using StandardScaler and feature importance analysis using permutation importance.

## Model Performance

The initial recorded AI4I test results are:

| Model | Accuracy | Failure Recall | Failure F1 |
|---|---|---|---|
| SVC + SMOTE | 79.62% | 85% | 0.25 |
| Logistic Regression + SMOTE | 83.12% | 78% | 0.27 |
| Tuned SVC | 80.62% | 85% | 0.26 |

These metrics are recorded notebook outputs from a test set containing 1,001 observations.

Because machine failures are relatively rare, overall accuracy alone is insufficient to evaluate model quality.

Precision, recall, and F1-score for the failure class are particularly important.

## Additional Experiments

The notebook also includes:

- Confusion matrix analysis
- Accuracy and F1-score comparisons
- Permutation-based feature importance
- Feature selection and model retraining
- Comparisons of balanced and imbalanced training approaches
- Hyperparameter optimization
- Experiments using synthetic and example classification datasets

Results from additional datasets should not be interpreted as AI4I failure prediction results.

## Project Structure

```text
ML-ariza-analiz-ai4i/
├── ML-ariza-analiz-ai4i.ipynb
└── ai4i2020.csv
```

## Installation and Usage

### Requirements

- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

### Installation

Clone the repository:

```bash
git clone https://github.com/sumeyyeduz35/ML-ariza-analiz-ai4i.git
cd ML-ariza-analiz-ai4i
```

Install the required dependencies:

```bash
python -m pip install notebook pandas numpy scikit-learn imbalanced-learn matplotlib
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `ML-ariza-analiz-ai4i.ipynb` to explore the analysis.

The notebook includes multiple experiments. Check dataset assignments and variable dependencies before executing its cells.

## Technical Limitations

This is an experimental machine learning analysis rather than a production-ready predictive maintenance system.

- The primary analysis uses a subset of the original dataset.
- Class imbalance affects model performance.
- Some feature selection steps use information from the test set.
- The SMOTE and cross-validation workflow could be improved to avoid potential data leakage.
- Later notebook sections include separate synthetic and example datasets.
- The reported metrics should not be considered independently validated production performance.

## Project Status

**Completed — Experimental machine learning analysis.**

The project demonstrates data preprocessing, classification, class imbalance handling, model comparison, and performance evaluation techniques.
