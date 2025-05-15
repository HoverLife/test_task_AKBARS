import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Загрузка и разбиение данных
df = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')
X = df.drop(['ИД','ДЕФОЛТ60'], axis=1)
y = df['ДЕФОЛТ60']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2. Pipeline: SMOTE + Random Forest (баланс весов) + калибровка
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 3. Поиск гиперпараметров (Randomized)
param_dist = {
    'rf__n_estimators': [100, 200, 500],
    'rf__max_depth': [5, 10, 20, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['auto', 'sqrt', 0.5]
}
rs = RandomizedSearchCV(
    pipe, param_dist, n_iter=30, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
rs.fit(X_train, y_train)
best_pipe = rs.best_estimator_
print('Best RF params:', rs.best_params_)

# 4. Platt scaling на SMOTE-resampled
X_s, y_s = SMOTE(random_state=42).fit_resample(X_train, y_train)
clf = CalibratedClassifierCV(best_pipe.named_steps['rf'], method='sigmoid', cv=5)
clf.fit(X_s, y_s)

# 5. Валидация: предсказание и оптимизация threshold с помощью F1
y_pred_proba = clf.predict_proba(X_val)[:,1]
thresholds = np.linspace(0.01,0.99,99)
best_thr, best_f1 = 0.5, 0
for t in thresholds:
    f1 = f1_score(y_val, (y_pred_proba>t).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, t
print(f'Optimal threshold: {best_thr:.2f}, F1={best_f1:.3f}')

# 6. Метрики


y_pred = (y_pred_proba > best_thr).astype(int)
metrics = {
    'AUC': roc_auc_score(y_val, y_pred_proba),
    'Accuracy': accuracy_score(y_val, y_pred),
    'Precision': precision_score(y_val, y_pred),
    'Recall': recall_score(y_val, y_pred),
    'F1': best_f1,
    'gini' : 2 * roc_auc_score(y_val, y_pred_proba) - 1
}

# gini = 2 * metrics['AUC'] - 1
#     metrics['Gini'] = gini
#     print(f"Gini coefficient: {gini:.3f}")

cm = confusion_matrix(y_val, y_pred)
print('[Final RF] ' + ', '.join([f"{k}={v:.3f}" for k,v in metrics.items()]))
print('Confusion matrix:', cm)

# 7. KS 
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
ks = max(tpr - fpr)
print(f'KS statistic: {ks:.3f}')

# 8. ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={metrics['AUC']:.3f}, KS={ks:.3f}")
textstr = '\n'.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
plt.gca().text(0.55, 0.2, textstr, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Final RF ROC Curve'); plt.legend(); plt.tight_layout()
plt.savefig('./imgs/rf_metrics_roc.png'); plt.show()

# 9. Сохранение
df_test = pd.read_excel('./data/data_previous.xls', sheet_name='Тестовая')
Xt = df_test.drop(['ИД','ДЕФОЛТ60 (Прогноз)','Скоринговый балл'], axis=1)
proba_test = clf.predict_proba(Xt)[:,1]
pred_test = (proba_test > best_thr).astype(int)
out = pd.DataFrame({
    'ИД': df_test['ИД'],
    'ДЕФОЛТ60 (Прогноз)': pred_test,
    'Скоринговый балл': proba_test
})
out.to_csv('./results/results_rf_BEST_MODEL.csv', index=False, encoding='utf-8-sig')