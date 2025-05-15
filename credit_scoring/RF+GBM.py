import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, average_precision_score,
    brier_score_loss, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# 1) Загрузка и разбиение данных
df = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')
X = df.drop(['ИД','ДЕФОЛТ60'], axis=1)
y = df['ДЕФОЛТ60']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2) Балансировка классов
sm = SMOTE(random_state=42)
X_s, y_s = sm.fit_resample(X_train, y_train)

# 3) Определение базовых моделей с калибровкой
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf_rf = CalibratedClassifierCV(rf, cv=5, method='sigmoid')
clf_rf.fit(X_s, y_s)

gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                     max_depth=None, random_state=42)
clf_gbm = CalibratedClassifierCV(gbm, cv=5, method='sigmoid')
clf_gbm.fit(X_s, y_s)

# 4) StackingClassifier
estimators = [
    ('rf', clf_rf),
    ('gbm', clf_gbm)
]
meta = LogisticRegression(solver='liblinear', random_state=42)
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
stack.fit(X_s, y_s)

# 5) Прогноз на валидации
proba_val = stack.predict_proba(X_val)[:,1]
pred_val = stack.predict(X_val)

# 6) Расчет метрик
metrics = {
    'AUC': roc_auc_score(y_val, proba_val),
    'Accuracy': accuracy_score(y_val, pred_val),
    'Precision': precision_score(y_val, pred_val),
    'Recall': recall_score(y_val, pred_val),
    'F1': f1_score(y_val, pred_val),
    'Avg Precision (PR AUC)': average_precision_score(y_val, proba_val),
    'Brier Score': brier_score_loss(y_val, proba_val)
}
print("Stacked Model Metrics:")
for k, v in metrics.items(): print(f"{k}: {v:.3f}")
print("Classification Report:", classification_report(y_val, pred_val))
print("Confusion Matrix:", confusion_matrix(y_val, pred_val))

# 7) Визуализация ROC и PR кривых
fpr, tpr, _ = roc_curve(y_val, proba_val)
precision, recall, _ = precision_recall_curve(y_val, proba_val)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f'AUC={metrics["AUC"]:.3f}')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Stacked ROC Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f'AP={metrics["Avg Precision (PR AUC)"]:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('./imgs/RF+GBM_roc_pr.png')
plt.show()

# # 8) Калибровочная кривая
# prob_true, prob_pred = calibration_curve(y_val, proba_val, n_bins=10)
# plt.figure(figsize=(6,6))
# plt.plot(prob_pred, prob_true, marker='o')
# plt.plot([0,1],[0,1],'--')
# plt.xlabel('Predicted Probability'); plt.ylabel('True Probability')
# plt.title('Calibration Curve')
# plt.tight_layout()
# plt.savefig('./imgs/stacked_calibration.png')
# plt.show()

# 9) Сохранение прогнозов для теста
df_test = pd.read_excel('./data/data_previous.xls', sheet_name='Тестовая')
Xt = df_test.drop(['ИД','ДЕФОЛТ60 (Прогноз)','Скоринговый балл'], axis=1)
proba_test = stack.predict_proba(Xt)[:,1]
pred_test = stack.predict(Xt)
out = pd.DataFrame({
    'ИД': df_test['ИД'],
    'ДЕФОЛТ60 (Прогноз)': pred_test,
    'Скоринговый балл': proba_test
})
out.to_csv('./results/results_RF+GBM.csv', index=False, encoding='utf-8-sig')