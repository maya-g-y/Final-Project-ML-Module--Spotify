# 🎧 Spotify Genre Classification using Machine Learning

## 📘 Overview
This project builds a **machine learning pipeline** that predicts a song’s **playlist genre** based on its audio and metadata features from the **Spotify Songs dataset**.  
The workflow follows the complete ML lifecycle — from **data preparation** and **exploratory analysis**, through **feature engineering and selection**, to **model training, tuning, and evaluation**.

---

## 📂 Project Structure
| Stage | File | Description |
|--------|------|-------------|
| 1️⃣ Data Preparation | `spotify1_data_prep.py` | Loads raw Kaggle data, cleans text, extracts release year/month, reduces rare categories, and saves as `spotify_flat_file.pkl`. |
| 2️⃣ Exploratory Data Analysis (EDA) | `spotify2_eda.py` | Performs descriptive statistics, missing value checks, distribution plots, correlation & Kruskal–Wallis tests, and saves `final_df_EDA.pkl`. |
| 3️⃣ Data Cleansing | `spotify3_data_cleansing.py` | Detects and handles outliers using the **IQR method** and applies **selective Winsorization** while preserving correlation structure. Saves `final_df_cleansed.pkl`. |
| 4️⃣ Feature Engineering & Selection | `spotify4_feature_engineering_&_selection (1).py` | Creates interaction, ratio, and temporal features; scales numeric values; applies **ANOVA + L1**, **model-committee**, and **union/intersection** selection methods. Produces `X_train_final_full.pkl` and `y_train.pkl`. |
| 5️⃣ Model Selection & Fine-Tuning | `spotify5_model_selection_and_fine_tuning.py` | Trains and tunes multiple classifiers (Logistic, SVM, RF, GB, AdaBoost, XGBoost), performs **GridSearchCV**, and selects the final XGBoost model with early stopping. Evaluates on the test set and plots performance metrics. |

---

## 🎯 Project Objective
> **Research Question:**  
> *Which musical features most strongly differentiate between Spotify genres, and can we accurately predict genre from them?*

---

## 🧰 Tech Stack
- **Language:** Python  
- **Environment:** Google Colab  
- **Core Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`, `scipy`  
- **Storage:** Google Drive (Pickle-based pipeline between stages)  
- **Data Source:** [Kaggle – 30,000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

---

## ⚙️ Pipeline Summary

### 1. Data Preparation
- Cleaned text fields, normalized casing and punctuation.  
- Extracted `release_year` and `release_month`.  
- Reduced rare categories in text-based fields (`playlist_name`, `track_artist`) to `"other"`.  
- Saved the cleaned dataset as `spotify_flat_file.pkl`.

### 2. Exploratory Data Analysis
- Analyzed genre/subgenre distributions and relationships.  
- Generated violin and boxplots for audio features across genres.  
- Identified strong correlations:
  - `energy ↔ loudness` (positive)  
  - `acousticness ↔ energy` (negative)  
  - `danceability ↔ valence` (positive)  
- Conducted **Spearman correlation** and **Kruskal–Wallis** tests to confirm feature–genre significance.  
- Result: all numerical features significantly differ across genres.

### 3. Data Cleansing & Outlier Treatment
- Detected outliers using the **IQR** method per feature.  
- Assessed impact on feature distribution and correlation with the target (`playlist_genre`).  
- Applied **Winsorization** only to variables where outliers distorted distributions without changing correlations (e.g. `acousticness`, `liveness`, `duration_ms`, `loudness`, `tempo`).  
- Verified that correlations before/after treatment remained stable.

### 4. Feature Engineering & Selection
- Engineered new numeric and interaction features:
  - Ratios (e.g. `energy_ratio`, `vocal_focus`)
  - Temporal features (`song_age`, `release_decade`, seasonal flags)
  - Composite features (`mood_index`, `complexity`)  
- Scaled numeric variables using **StandardScaler**.  
- Performed **feature selection** via:
  1. **ANOVA F-test** + **L1 Logistic Regression** (statistical + model-based)
  2. **Model Committee Voting** (Logistic L1, SVM L1, GradientBoost, RandomForest)
  3. **Union vs Intersection** comparison.  
- Final selected feature set (`Union`) achieved the best F1-score (≈0.59).

### 5. Model Selection & Fine-Tuning
- Split dataset into **Train / Validation / Test** (≈64% / 16% / 20%).  
- Ran **GridSearchCV** for:
  - Logistic Regression (L1/L2)
  - LinearSVC
  - RandomForest
  - GradientBoost
  - AdaBoost
  - XGBoost (with early stopping)  
- Evaluated all via **Macro F1** and **Accuracy**.  
- Selected final **XGBoost** model:  
  `max_depth=5`, `learning_rate=0.08`, `colsample_bytree=0.7`, `subsample=0.8`, `gamma=0.25`.  
- Performance:
  - **Validation Macro-F1:** 0.59  
  - **Test Macro-F1:** 0.60  
  - **Top Genres:** Rock (0.77), EDM (0.70), Rap (0.66)  
  - **Weaker:** Pop (0.44), Latin (0.51), R&B (0.51)

---

## 📊 Results Summary
| Dataset | Accuracy | Macro F1 | Notes |
|----------|-----------|-----------|-------|
| Validation | 0.59 | 0.59 | Strong balance across genres |
| Test | 0.60 | 0.60 | Confirms generalization |
| Top Predictive Features | `energy`, `loudness`, `valence`, `danceability`, `release_year`, `mood_index` |

---

## 🧠 Key Insights
- Audio features are **statistically significant** differentiators of genres.  
- **Energy, loudness, and valence** emerge as the most genre-discriminative.  
- Interaction and ratio features improved model interpretability and performance.  
- Ensemble feature voting provided a stable, compact feature subset.  
- **XGBoost** offered the best trade-off between interpretability, speed, and accuracy.

---

## 🚀 How to Run
1. Clone the repository and open in **Google Colab** or any Jupyter environment.  
2. Run scripts in sequential order (`spotify1` → `spotify5`).  
3. Ensure `/content/drive/MyDrive/pickle_files/` exists in your Drive for saving intermediate data.  
4. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost seaborn matplotlib scipy kagglehub openpyxl
5. The final trained model and predictions are saved as:
   ```bash
   best_XGBoost_CV.joblib
   xgb_test_predictions.csv

## 🏁 Conclusion
The project demonstrates a **complete supervised ML pipeline** capable of identifying musical genre patterns from numerical and metadata features. 
Despite genre overlap challenges, the **XGBoost classifier** achieved balanced performance and meaningful interpretability, laying a solid foundation for further work such as deep audio embeddings or lyric-based analysis.

