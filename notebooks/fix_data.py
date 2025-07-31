import pandas as pd
from sklearn.model_selection import train_test_split

# 重新加载最原始的数据，确保有customerID
original_df = pd.read_csv(r'C:\Users\Administrator\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 重新执行一遍预处理，以获得和之前完全一样的索引
processed_df = original_df.copy()
processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce').fillna(0)
processed_df['Churn'] = processed_df['Churn'].map({'Yes': 1, 'No': 0})
processed_df_dummied = pd.get_dummies(processed_df.drop('customerID', axis=1), drop_first=True)
x = processed_df_dummied.drop('Churn', axis=1)
y = processed_df_dummied['Churn']

# 使用完全相同的random_state，确保我们得到和上次一模一样的测试集索引
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 这就是关键！根据测试集的索引，从原始数据中找到对应的customerID
customer_ids_for_test_set = original_df.loc[x_test.index, 'customerID']

# 加载我们已经生成好的Results文件
results_df = pd.read_csv(r'C:\Users\Administrator\Desktop\Telco_Churn_Results_Data.csv')

# 把customerID加进去！
# .reset_index(drop=True) 是为了确保两个表能正确拼接
results_df['customerID'] = customer_ids_for_test_set.reset_index(drop=True)

# 调整列的顺序，让customerID在第一列，更专业！
cols = ['customerID'] + [col for col in results_df if col != 'customerID']
results_df = results_df[cols]

# 用我们完美的新版本，覆盖掉旧文件
output_path = r'C:\Users\Administrator\Desktop\Telco_Churn_Results_Data.csv'
results_df.to_csv(output_path, index=False)

print(f"--- 数据修复完成！ ---")
print(f"新的 'Telco_Churn_Results_Data.csv' 已经包含了customerID，并保存到了你的桌面！")
