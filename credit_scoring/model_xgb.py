import pandas as pd
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')
X = df.drop(['ИД','ДЕФОЛТ60'], axis=1)
y = df['ДЕФОЛТ60']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2. SMOTE для балансировки
sm = SMOTE(random_state=42)
X_s, y_s = sm.fit_resample(X_train, y_train)

# 3. GridSearch для XGBoost
dtrain = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [100,200],
    'max_depth': [3,6],
    'learning_rate': [0.01, 0.1]
}
gs = GridSearchCV(dtrain, param_grid, cv=5, scoring='f1', n_jobs=-1)
gs.fit(X_s, y_s)
best = gs.best_estimator_
print('XGB Best params:', gs.best_params_)

# 4. Валидация и подбор порога
proba_val = best.predict_proba(X_val)[:,1]
best_thresh, best_f1 = 0.5, 0
from sklearn.metrics import f1_score
for t in [i/100 for i in range(1,100)]:
    f1 = f1_score(y_val, (proba_val>t).astype(int))
    if f1>best_f1: best_f1, best_thresh = f1, t
pred_val = (proba_val>best_thresh).astype(int)

# 5. Метрики
metrics = {
    'AUC': roc_auc_score(y_val, proba_val),
    'Accuracy': accuracy_score(y_val, pred_val),
    'Precision': precision_score(y_val, pred_val),
    'Recall': recall_score(y_val, pred_val),
    'F1': best_f1
}
cm = confusion_matrix(y_val, pred_val)
print('[XGB] ' + ', '.join([f"{k}={v:.3f}" for k,v in metrics.items()]))
print('Confusion matrix:', cm)

# 6. ROC + метрики
fpr, tpr, _ = roc_curve(y_val, proba_val)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={metrics['AUC']:.3f}")
textstr = ''.join([f"{k}={v:.3f}" for k,v in metrics.items()])
plt.gca().text(0.6, 0.2, textstr, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('XGB ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./imgs/xgb_metrics_roc.png')
plt.show()

# 7. Сохранение прогнозов на тестовой
df_test = pd.read_excel('./data/data_previous.xls', sheet_name='Тестовая')
Xt = df_test.drop(['ИД','ДЕФОЛТ60 (Прогноз)','Скоринговый балл'], axis=1)
proba_test = best.predict_proba(Xt)[:,1]
pred_test = (proba_test>best_thresh).astype(int)
out = pd.DataFrame({
    'ИД': df_test['ИД'],
    'ДЕФОЛТ60 (Прогноз)': pred_test,
    'Скоринговый балл': proba_test
})
out.to_csv('./results/results_xgb.csv', index=False, encoding='utf-8-sig')