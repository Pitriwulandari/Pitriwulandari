# Hasil Eksperimen Ghani

## Progress
- **Baseline**: 49.38%
- **Previous Best**: 54.06% (CatBoost + LightGBM blending)
- **Current Best**: 54.53% (Calibrated CatBoost Isotonic / Advanced Ensemble 3-model)
- **Current Target**: 70-80%

## Improvements yang Diterapkan (Latest Version):
1. **Advanced Feature Engineering** - rolling statistics, trend analysis, polynomial features
2. **Domain-Specific Features** - consistency metrics, improvement rate, academic potential score, learning momentum
3. **NEW: Advanced Features** - percentile-based features, skewness/kurtosis, momentum indicators, composite scores
4. **SMOTE** - untuk handle class imbalance
5. **Multiple Model Alternatives** - Random Forest, LightGBM, CatBoost, Gradient Boosting, XGBoost, ExtraTrees
6. **CatBoost-Specific Hyperparameter Tuning** - RandomizedSearchCV dengan 50 iterasi
7. **NEW: Advanced Ensemble** - Multi-model blending dengan 6+ models
8. **NEW: Neural Network** - MLPClassifier dengan berbagai architectures
9. **NEW: Advanced Feature Selection** - Mutual Information based selection
10. **Robust Cross-Validation** - Repeated Stratified K-Fold (5 splits, 3 repeats)

## Hasil Eksperimen Terbaru (dengan Advanced Features):

### Individual Models:
- Random Forest: 47.03%
- LightGBM: 51.41%
- **CatBoost: 54.37%** (best single model)
- Gradient Boosting: 47.50%
- XGBoost: 49.84%
- ExtraTrees: 45.16%

### Ensemble Methods:
- Voting Classifier (4 models): 52.66%
- Stacking Classifier: 52.50%
- Weighted Voting (CatBoost=0.8, LightGBM=0.2): 54.22%
- **Advanced 3-model Blending (CatBoost=0.7, LightGBM=0.2, XGBoost=0.1): 54.53%**

### Calibration:
- **Calibrated CatBoost (Isotonic): 54.53%** ⭐
- Calibrated CatBoost (Sigmoid): 53.91%

### Other Techniques:
- CatBoost Tuned: 53.44%
- CatBoost CV Mean: 53.32%
- Neural Network (MLP): 52.34%
- Feature Selection (MI): 51.25% (menurunkan akurasi)

## Analysis:
- **Improvement dari baseline**: +5.15% (49.38% → 54.53%)
- **Improvement dari previous best**: +0.47% (54.06% → 54.53%)
- **Feature selection tidak membantu** - menurunkan akurasi
- **Calibration memberikan improvement kecil**
- **Advanced ensemble memberikan hasil terbaik**

## Gap ke Target:
- **Target**: 70-80%
- **Current**: 54.53%
- **Gap**: 15.47-25.47 poin

## Next Steps untuk Mencapai 70-80%:
1. Deep Learning dengan TensorFlow/PyTorch (CNN/RNN untuk time-series)
2. Hyperparameter optimization dengan Optuna (more extensive)
3. Cross-validation dengan berbagai strategies (TimeSeriesSplit, GroupKFold)
4. Data augmentation techniques (SMOTE variants, ADASYN)
5. Meta-learning / stacking dengan lebih banyak base learners
6. Feature importance analysis untuk remove noisy features
7. Target encoding untuk categorical features
8. Time-series specific models (Prophet, ARIMA-based features)