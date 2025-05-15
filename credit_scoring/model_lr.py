import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

# 1) Загрузка обучающей
df_train = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')
X = df_train.drop(['ИД','ДЕФОЛТ60'], axis=1)
y = df_train['ДЕФОЛТ60']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 2) Пайплайн и GridSearch
pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('kbins', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')),
    ('woe', ce.WOEEncoder()),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(solver='liblinear', random_state=42))
])
param_grid = {'lr__C': [0.1,1,10], 'lr__class_weight': [None,'balanced']}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
gs.fit(X_train, y_train)
best = gs.best_estimator_

# 3) Валидация и подбор порога по F1
proba_val = best.predict_proba(X_val)[:,1]
best_thresh, best_f1 = 0.5, 0
from sklearn.metrics import f1_score
for t in [i/100 for i in range(1,100)]:
    f1 = f1_score(y_val, (proba_val>t).astype(int))
    if f1>best_f1: best_f1, best_thresh = f1,t
pred_val = (proba_val>best_thresh).astype(int)

# 4) Расчёт метрик и ROC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, roc_curve
)
auc = roc_auc_score(y_val, proba_val)
acc = accuracy_score(y_val, pred_val)
prec = precision_score(y_val, pred_val)
rec = recall_score(y_val, pred_val)
f1 = best_f1
cm = confusion_matrix(y_val, pred_val)
print(f"[LR] AUC={auc:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
print("Confusion matrix:\n",cm)

fpr,tpr,_=roc_curve(y_val,proba_val)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'AUC={auc:.3f}')
plt.text(0.6,0.2,f"Acc={acc:.3f}\nPrec={prec:.3f}\nRec={rec:.3f}\nF1={f1:.3f}",bbox=dict(facecolor='white',alpha=0.8))
plt.xlabel('FPR');plt.ylabel('TPR');plt.title('LR ROC Curve');plt.legend();plt.tight_layout()
plt.savefig('./imgs/lr_metrics_roc.png');plt.show()

# 5) Сохранение прогнозов на тестовой
df_test = pd.read_excel('./data/data_previous.xls', sheet_name='Тестовая')
Xt = df_test.drop(['ИД','ДЕФОЛТ60 (Прогноз)','Скоринговый балл'], axis=1)
proba_test = best.predict_proba(Xt)[:,1]
pred_test = (proba_test>best_thresh).astype(int)
out = pd.DataFrame({
    'ИД': df_test['ИД'],
    'ДЕФОЛТ60 (Прогноз)': pred_test,
    'Скоринговый балл': proba_test
})
out.to_csv('./results/results_lr.csv', index=False, encoding='utf-8-sig')