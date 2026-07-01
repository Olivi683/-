import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ================== 1. 加载数据 ==================
data = joblib.load('processed_data.pkl')   # 修改为你的路径
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

# 仅删除两个直接泄露特征
safe_blacklist = ['Crisis_Severity_Index', 'Mental_Health_Composite']
drop_idx = [i for i, f in enumerate(feature_names) if f in safe_blacklist]
X_train_clean = np.delete(X_train, drop_idx, axis=1)
X_test_clean = np.delete(X_test, drop_idx, axis=1)
clean_features = [f for f in feature_names if f not in safe_blacklist]
print(f"保留特征数量: {len(clean_features)}")

# ================== 2. 模型训练（最终调优） ==================
RANDOM_STATE = 42

# 随机森林（最强配置）
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_clean, y_train)

# SVM（中等正则化）
svm = SVC(kernel='rbf', C=0.4, probability=True, random_state=RANDOM_STATE)
svm.fit(X_train_clean, y_train)

# 逻辑回归（强正则化）
lr = LogisticRegression(max_iter=2000, C=0.3, class_weight='balanced',
                        random_state=RANDOM_STATE)
lr.fit(X_train_clean, y_train)

# ================== 3. 评估 ==================
def evaluate(name, y_true, y_pred, y_proba):
    return {
        '模型': name,
        '准确率': accuracy_score(y_true, y_pred),
        '精确率': precision_score(y_true, y_pred, zero_division=0),
        '召回率': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_proba),
        '平衡准确率': balanced_accuracy_score(y_true, y_pred)
    }

results = pd.DataFrame([
    evaluate('随机森林', y_test, rf.predict(X_test_clean), rf.predict_proba(X_test_clean)[:,1]),
    evaluate('SVM', y_test, svm.predict(X_test_clean), svm.predict_proba(X_test_clean)[:,1]),
    evaluate('逻辑回归', y_test, lr.predict(X_test_clean), lr.predict_proba(X_test_clean)[:,1])
])

print("\n===== 🎯 最终模型性能（随机森林最优）=====")
print(results.to_string(index=False))

# 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(rf, X_train_clean, y_train, cv=skf, scoring='f1')
print(f"\n随机森林 5折 CV F1 均值: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 特征重要性 Top-10
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
print("\n===== 随机森林 Top-10 重要特征 =====")
for i in range(10):
    print(f"{i+1:2}. {clean_features[indices[i]]:35s} {importances[indices[i]]:.4f}")

# Agent 决策阈值
from sklearn.model_selection import cross_val_predict
y_train_proba = cross_val_predict(rf, X_train_clean, y_train, cv=5, method='predict_proba')[:, 1]
thresholds = np.linspace(0.1, 0.9, 50)
best_t, best_f1 = 0.5, 0
for t in thresholds:
    preds = (y_train_proba >= t).astype(int)
    f1 = f1_score(y_train, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print(f"\nAgent 最优决策阈值: {best_t:.2f} (CV F1: {best_f1:.4f})")
