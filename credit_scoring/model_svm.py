import pandas as pd
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

# 1. Загрузка и разбиение данных
df = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')
X = df.drop(['ИД','ДЕФОЛТ60'], axis=1)
y = df['ДЕФОЛТ60']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2. Pipeline с SMOTE, масштабированием и SVM
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])
# 3. GridSearch по гиперпараметрам SVM
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
gs.fit(X_train, y_train)
best = gs.best_estimator_
print('SVM Best params:', gs.best_params_)

# 4. Валидация и подбор оптимального порога по F1
proba_val = best.predict_proba(X_val)[:,1]
best_thresh, best_f1 = 0.5, 0
from sklearn.metrics import f1_score
for t in [i/100 for i in range(1,100)]:
    f1 = f1_score(y_val, (proba_val>t).astype(int))
    if f1>best_f1: best_f1, best_thresh = f1, t
pred_val = (proba_val>best_thresh).astype(int)

# 5. Расчет метрик
metrics = {}
metrics['AUC'] = roc_auc_score(y_val, proba_val)
metrics['Accuracy'] = accuracy_score(y_val, pred_val)
metrics['Precision'] = precision_score(y_val, pred_val)
metrics['Recall'] = recall_score(y_val, pred_val)
metrics['F1'] = best_f1
cm = confusion_matrix(y_val, pred_val)
print('[SVM] ' + ', '.join([f"{k}={v:.3f}" for k,v in metrics.items()]))
print('Confusion matrix:', cm)

# 6. ROC + метрики
fpr, tpr, _ = roc_curve(y_val, proba_val)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={metrics['AUC']:.3f}")
textstr = ''.join([f"{k}={v:.3f}" for k,v in metrics.items()])
plt.gca().text(0.6, 0.2, textstr, bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./imgs/svm_metrics_roc.png')
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
out.to_csv('./results/results_svm.csv', index=False, encoding='utf-8-sig')