# Next Steps untuk Improve Akurasi (dari 53.75% ke 60-65%)

## Current Best Result
- **CatBoost single model**: 53.75% validation accuracy
- **Voting Classifier**: 51.25%
- **Baseline (sebelum improvement)**: 49.38%
- **Improvement**: +4.37%

## Masalah yang Terdeteksi

### 1. Ensemble Tidak Efektif
- Voting dan Stacking classifier tidak improve dibanding CatBoost single
- Kemungkinan karena model-model lain (RF, XGBoost, LightGBM) performanya lebih rendah

### 2. Hyperparameter Tuning Overfitting
- XGBoost tuned: CV 53.77% vs Validation 49.22%
- Gap 4.55% menunjukkan overfitting
- Perlu regularisasi lebih kuat atau cross-validation yang lebih robust

### 3. Feature Selection Tidak Membantu
- SelectKBest dan RFE menurunkan akurasi
- Semua fitur yang dibuat sudah relevan

## Strategi Improve Selanjutnya

### Priority 1: Focus pada CatBoost (Model Terbaik)
```python
# Hyperparameter tuning khusus untuk CatBoost
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist_cat = {
    'iterations': randint(200, 600),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.2),
    'l2_leaf_reg': uniform(1, 10),
    'border_count': randint(32, 256),
    'bagging_temperature': uniform(0, 1),
    'random_strength': uniform(0, 1),
    'one_hot_max_size': randint(2, 10)
}

cat_search = RandomizedSearchCV(
    CatBoostClassifier(random_state=42, verbose=False),
    param_distributions=param_dist_cat,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
```

### Priority 2: Cross-Validation yang Lebih Robust
```python
# Gunakan Repeated Stratified K-Fold untuk mengurangi variance
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_scores = cross_val_score(cat, X_train_smote, y_train_smote, cv=rskf, scoring='accuracy')
print(f"Mean CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

### Priority 3: Feature Engineering Lebih Domain-Specific
```python
# Fitur yang lebih spesifik untuk konteks pendidikan
def create_domain_features(df):
    df = df.copy()
    
    # 1. Consistency metrics
    df['nilai_consistency'] = 1 - (df['nilai_std'] / (df['nilai_mean'] + 1e-6))
    
    # 2. Improvement rate
    df['nilai_improvement_rate'] = (df['nilai_minggu_12'] - df['nilai_minggu_01']) / 12
    
    # 3. Activity engagement score
    aktivitas_cols = [col for col in df.columns if 'aktivitas_hari' in col]
    df['activity_engagement'] = df[aktivitas_cols].gt(df[aktivitas_cols].mean(axis=1)).sum(axis=1) / 16
    
    # 4. Performance trend (positive/negative)
    df['performance_trend'] = np.where(df['nilai_trend'] > 0, 1, 0)
    
    # 5. Task completion efficiency
    df['task_efficiency'] = df['rasio_tugas'] * df['aktivitas_mean']
    
    # 6. Academic potential score
    df['academic_potential'] = (
        df['skor_tryout'] * 0.4 + 
        df['total_skor'] * 0.3 + 
        df['nilai_mean'] * 0.3
    )
    
    return df
```

### Priority 4: Coba Neural Network (TabNet)
```python
# TabNet sering memberikan hasil baik untuk tabular data
from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    cat_idxs=[],
    cat_dims=[],
    lambda_sparse=1e-3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax'
)

clf.fit(
    X_train_smote.values, y_train_smote.values,
    eval_set=[(X_val.values, y_val.values)],
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=128
)
```

### Priority 5: Ensemble Hanya Model-Model Terbaik
```python
# Ensemble hanya CatBoost + LightGBM (2 model terbaik)
voting_top2 = VotingClassifier(
    estimators=[
        ('cat', CatBoostClassifier(iterations=300, depth=8, learning_rate=0.1, random_state=42, verbose=False)),
        ('lgbm', LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1))
    ],
    voting='soft',
    weights=[0.6, 0.4]  # Beri lebih banyak weight ke CatBoost
)
```

### Priority 6: Post-Processing / Calibration
```python
# Calibrate probabilities untuk improve prediction
from sklearn.calibration import CalibratedClassifierCV

calibrated_cat = CalibratedClassifierCV(
    cat, 
    method='isotonic', 
    cv=5
)
calibrated_cat.fit(X_train_smote, y_train_smote)
```

## Action Plan (Urutan Eksekusi)

1. **Langkah 1**: Hyperparameter tuning khusus untuk CatBoost
2. **Langkah 2**: Tambah domain-specific features
3. **Langkah 3**: Coba TabNet atau neural network sederhana
4. **Langkah 4**: Ensemble hanya 2 model terbaik dengan weighted voting
5. **Langkah 5**: Calibration dan post-processing

## Target Realistis
- **Current**: 53.75%
- **Short-term (dengan CatBoost tuning)**: 55-57%
- **Medium-term (dengan domain features)**: 58-62%
- **Long-term (dengan TabNet/ensemble)**: 62-65%

## Catatan Penting
- Jangan gunakan feature selection karena menurunkan akurasi
- Fokus pada CatBoost karena sudah terbukti terbaik
- Ensemble hanya efektif jika model-model yang di-ensemble performanya seimbang
- Cross-validation yang robust penting untuk menghindari overfitting
