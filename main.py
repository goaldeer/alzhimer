import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')

loc = "/content/drive/MyDrive/ai_deep/"

# 1. 데이터 로드
activity_df = pd.read_csv(loc + "train_activity.csv")
sleep_df = pd.read_csv(loc + "train_sleep.csv")
label_df = pd.read_csv(loc + "training_label.csv")
mmse_df = pd.read_csv(loc + "train_mmse.csv")

val_activity = pd.read_csv(loc + "val_activity.csv")
val_sleep = pd.read_csv(loc + "val_sleep.csv")
val_label = pd.read_csv(loc + "val_label.csv")
val_mmse = pd.read_csv(loc + "val_mmse.csv")

# 2. 이상치 처리 함수
def clean_outliers(df, numeric_cols):
    z_scores = np.abs(zscore(df[numeric_cols]))
    df[numeric_cols] = df[numeric_cols].mask(z_scores > 2)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

numeric_cols_activity = activity_df.select_dtypes(include=np.number).columns
numeric_cols_sleep = sleep_df.select_dtypes(include=np.number).columns

activity_df = clean_outliers(activity_df, numeric_cols_activity)
sleep_df = clean_outliers(sleep_df, numeric_cols_sleep)
val_activity = clean_outliers(val_activity, numeric_cols_activity)
val_sleep = clean_outliers(val_sleep, numeric_cols_sleep)

# 3. Approximate Entropy 계산 함수
def compute_apen(U, m=2, r=None):
    U = np.array(U)
    N = len(U)
    if r is None:
        r = 0.3 * np.std(U)
    if N <= m + 1:
        return 0
    def _phi(m):
        x = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(np.log(C + 1e-10)) / (N - m + 1)
    return abs(_phi(m) - _phi(m + 1))

apen_cols_activity = ["CONVERT(activity_class_5min USING utf8)", "CONVERT(activity_met_1min USING utf8)"]
apen_cols_sleep = ["CONVERT(sleep_hr_5min USING utf8)", "CONVERT(sleep_hypnogram_5min USING utf8)", "CONVERT(sleep_rmssd_5min USING utf8)"]

def compute_apen_by_email(df, email_col, target_cols):
    apen_df = pd.DataFrame()
    apen_df['EMAIL'] = df[email_col].unique()
    for col in target_cols:
        apen_values = []
        for email in apen_df['EMAIL']:
            series = df[df[email_col] == email][col].dropna()
            try:
                series = list(map(float, str(series.values[0]).split('/')))
            except:
                series = []
            apen = compute_apen(series) if len(series) > 10 else 0
            apen_values.append(apen)
        apen_df[f'ApEn_{col}'] = apen_values
    return apen_df

apen_act_df = compute_apen_by_email(activity_df, 'EMAIL', apen_cols_activity)
apen_sleep_df = compute_apen_by_email(sleep_df, 'EMAIL', apen_cols_sleep)
val_apen_act_df = compute_apen_by_email(val_activity, 'EMAIL', apen_cols_activity)
val_apen_sleep_df = compute_apen_by_email(val_sleep, 'EMAIL', apen_cols_sleep)

# 4. 피처 집계 및 병합
grouped_activity = activity_df.groupby("EMAIL")[numeric_cols_activity].mean().reset_index()
grouped_sleep = sleep_df.groupby("EMAIL")[numeric_cols_sleep].mean().reset_index()
mmse_small = mmse_df[['SAMPLE_EMAIL', 'TOTAL']].rename(columns={'SAMPLE_EMAIL': 'EMAIL', 'TOTAL': 'MMSE_TOTAL'})

val_grouped_activity = val_activity.groupby("EMAIL")[numeric_cols_activity].mean().reset_index()
val_grouped_sleep = val_sleep.groupby("EMAIL")[numeric_cols_sleep].mean().reset_index()
val_mmse_small = val_mmse[['SAMPLE_EMAIL', 'TOTAL']].rename(columns={'SAMPLE_EMAIL': 'EMAIL', 'TOTAL': 'MMSE_TOTAL'})

merged_df = grouped_activity.merge(grouped_sleep, on="EMAIL")
merged_df = merged_df.merge(mmse_small, on="EMAIL", how="left")
merged_df = merged_df.merge(apen_act_df, on="EMAIL", how="left")
merged_df = merged_df.merge(apen_sleep_df, on="EMAIL", how="left")
train_df = merged_df.merge(label_df, left_on="EMAIL", right_on="SAMPLE_EMAIL").drop(columns=["EMAIL", "SAMPLE_EMAIL"])

val_merged = val_grouped_activity.merge(val_grouped_sleep, on="EMAIL")
val_merged = val_merged.merge(val_mmse_small, on="EMAIL", how="left")
val_merged = val_merged.merge(val_apen_act_df, on="EMAIL", how="left")
val_merged = val_merged.merge(val_apen_sleep_df, on="EMAIL", how="left")
val_df = val_merged.merge(val_label, left_on="EMAIL", right_on="SAMPLE_EMAIL").drop(columns=["EMAIL", "SAMPLE_EMAIL"])

# 5. 라벨 인코딩 및 Mutual Information 기반 특징 선택
le = LabelEncoder()
y = le.fit_transform(train_df["DIAG_NM"])
X = train_df.drop(columns=["DIAG_NM"])

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_selected = pd.Series(mi_scores, index=X.columns)
mi_selected = mi_selected[mi_selected > 0].index.tolist()
X_selected = X[mi_selected]

val_X = val_df.drop(columns=["DIAG_NM"])[mi_selected]
y_val_true = le.transform(val_df["DIAG_NM"])

# 6. 스케일링
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)
val_scaled = scaler.transform(val_X)

# 7. 모델 생성 함수
def create_model(input_dim):
    model = Sequential([
        Dense(128, input_shape=(input_dim,)),
        LeakyReLU(0.1),
        Dropout(0.4),
        Dense(64),
        LeakyReLU(0.1),
        Dropout(0.4),
        Dense(len(np.unique(y)), activation='softmax')
    ])
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 8. 학습 및 평가 함수
def train_and_evaluate():
    X_train_resampled, y_train_resampled = X_scaled, y

    model = create_model(X_train_resampled.shape[1])
    model.fit(X_train_resampled, y_train_resampled, epochs=60, batch_size=64, verbose=0)

    y_val_pred = np.argmax(model.predict(val_scaled), axis=1)

    acc = accuracy_score(y_val_true, y_val_pred) * 100
    f1 = f1_score(y_val_true, y_val_pred, average='weighted') * 100

    print("===== Baseline (No Oversampling) =====")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Weighted F1-score: {f1:.2f}%")
    print("Classification Report:\n", classification_report(y_val_true, y_val_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_val_true, y_val_pred))

# 9. 실행
train_and_evaluate()
