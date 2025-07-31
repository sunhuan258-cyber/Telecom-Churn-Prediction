import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# 1. 加载和预处理数据（和之前完全一样）
file_path = r'C:\Users\Administrator\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv'
churn_df = pd.read_csv(file_path)
churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce')
churn_df['TotalCharges'] = churn_df['TotalCharges'].fillna(0)
processed_df = churn_df.drop('customerID', axis=1)
processed_df['Churn'] = processed_df['Churn'].map({'Yes': 1, 'No': 0})
processed_df = pd.get_dummies(processed_df, drop_first=True)
x = processed_df.drop('Churn', axis=1)
y = processed_df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 2. SMOTE处理
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# 3. 定义网格搜索（和之前完全一样）
param_grid = {
    'classifier__n_estimators': [200],
    'classifier__max_depth': [5],
    'classifier__learning_rate': [0.1]
}
pipeline_for_grid = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])
grid_search = GridSearchCV(estimator=pipeline_for_grid,
                         param_grid=param_grid,
                         cv=5,
                         scoring='roc_auc',
                         n_jobs=-1)

# 4. 训练模型（这是获取final_model的关键）
print("--- 正在重新训练最终模型... ---")
grid_search.fit(x_train_smote, y_train_smote)
final_model = grid_search.best_estimator_
print("---" + " 模型训练完成！ ---")

# 5. 保存模型！ (修正了路径问题)
model_path = r'C:\Users\Administrator\Desktop\telecom_churn_model.joblib'
joblib.dump(final_model, model_path)

print(f"\n--- 任务完成！---")
print(f"我们宝贵的模型已经成功保存到: {model_path}")
print("这把'宝剑'已经入鞘，随时可以展示给世界了！")