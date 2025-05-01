import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Importing different models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# 选择算法
model_choice = "XGBoost"  # "RandomForest", "XGBoost", or "LightGBM"

# 目标变量
forward_targets = ["Viscosity"]
back_targets = ['B2O3', 'SiO2', 'Al2O3', 'Na2O', 'BaO', 'TiO2', 'CeO2', 'K2O', 'CaO', 'MgO', 'Basicity']

# 读取数据
pred_data = pd.read_excel('D:\\Predict\\data.xlsx', header=None, sheet_name="data")
forward_data = pd.read_excel('D:\\Predict\\data.xlsx', header=None, sheet_name="forward").iloc[1:, 0:12]
back_data_full = pd.read_excel('D:\\Predict\\data.xlsx', header=None, sheet_name="back").iloc[1:, :]  # 先全读

# 根据back_targets列数动态提取
if len(back_targets) > back_data_full.shape[1]:
    raise ValueError(f"back_targets数量({len(back_targets)})大于实际数据列数({back_data_full.shape[1]})！请检查Excel文件！")
back_data = back_data_full.iloc[:, :len(back_targets)]  # 动态选择需要的列

# 切分数据集
switch = "反向"  # "正向" or "反向"
if switch == "正向":
    x = pred_data.iloc[1:, 0:12]
    y = pred_data.iloc[1:, 12]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

elif switch == "反向":
    x = pred_data.iloc[1:, 0:11]
    y = pred_data.iloc[1:, 11:13]
    x_train, x_test, y_train, y_test = train_test_split(back_data, pred_data.iloc[1:, 0:11], test_size=0.2, random_state=10)

else:
    raise ValueError("switch只能是 '正向' 或 '反向'！")

print("*************************** 选择的算法: ", model_choice, " ****************************")

# 根据算法选择模型
if model_choice == "RandomForest":
    if switch == "反向":
        model = MultiOutputRegressor(RandomForestRegressor(random_state=10))
    else:
        model = RandomForestRegressor(random_state=10)

elif model_choice == "XGBoost":
    if switch == "反向":
        model = MultiOutputRegressor(xgb.XGBRegressor(random_state=10))
    else:
        model = xgb.XGBRegressor(random_state=10)

elif model_choice == "LightGBM":
    if switch == "反向":
        model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=10))
    else:
        model = lgb.LGBMRegressor(random_state=10)

else:
    raise ValueError("model_choice只能是 'RandomForest', 'XGBoost', 或 'LightGBM'！")

# 训练模型
model.fit(x_train, y_train)

# 预测
y_pred = model.predict(x_test)

# 评估
print(f"{model_choice}结果如下：")
print("训练集分数：", model.score(x_train, y_train))
print("验证集分数：", model.score(x_test, y_test))

# 预测新数据并保存
if switch == "正向":
    result = model.predict(forward_data)
    pd.DataFrame(result, columns=forward_targets).to_excel('D:\\Predict\\forward_result.xlsx', index=False)

elif switch == "反向":
    result = model.predict(back_data)
    pd.DataFrame(result, columns=back_targets).to_excel('D:\\Predict\\back_result.xlsx', index=False)

print("预测并保存成功！")

# -------------------- 数据分布 --------------------
print("绘制数据分布直方图...")

if switch == "正向":
    plt.figure(figsize=(12, 8))
    plt.hist(y_train, bins=30, alpha=0.7, label='Train')
    plt.hist(y_test, bins=30, alpha=0.7, label='Test')
    plt.xlabel('Viscosity')
    plt.ylabel('Frequency')
    plt.title('Data Distribution of Viscosity')
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:\\Predict\\forward_distribution.png', dpi=300)
    plt.show()

elif switch == "反向":
    for i, target_name in enumerate(back_targets):
        plt.figure(figsize=(6, 4))
        plt.hist(y_train.iloc[:, i], bins=30, alpha=0.7, label='Train')
        plt.hist(y_test.iloc[:, i], bins=30, alpha=0.7, label='Test')
        plt.xlabel(target_name)
        plt.ylabel('Frequency')
        plt.title(f'Data Distribution of {target_name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'D:\\Predict\\back_distribution_{target_name}.png', dpi=300)
        plt.show()

# -------------------- Spearman's Correlation Matrix --------------------
print("绘制Spearman相关系数矩阵...")

if switch == "正向":
    full_data = x.copy()
    full_data['Viscosity'] = y
else:
    full_data = y.copy()
    full_data.columns = back_targets

corr = full_data.corr(method='spearman')

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
plt.title("Spearman's Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig('D:\\Predict\\spearman_corr_matrix.png', dpi=300)
plt.show()

# -------------------- 真实值 vs 预测值 + 拟合线 + 理想线 --------------------
print("绘制真实值 vs 预测值散点图 + 拟合线 + 理想线...")

fit_equations = []

if switch == "正向":
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='k', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    coef = np.polyfit(y_test, y_pred, 1)
    fit_line = np.poly1d(coef)
    x_fit = np.linspace(y_test.min(), y_test.max(), 100)
    plt.plot(x_fit, fit_line(x_fit), 'g-', label='Regression Fit')

    fit_equations.append(['Viscosity', f'y = {coef[0]:.4f}x + {coef[1]:.4f}'])

    plt.xlabel('Actual Viscosity')
    plt.ylabel('Predicted Viscosity')
    plt.title('Actual vs Predicted (Viscosity)')
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('D:\\Predict\\forward_scatter.png', dpi=300)
    plt.show()

elif switch == "反向":
    y_test_array = np.array(y_test)
    y_pred_array = np.array(y_pred)
    n_targets = len(back_targets)

    metrics_list = []

    for i in range(n_targets):
        r2 = r2_score(y_test_array[:, i], y_pred_array[:, i])
        mse = mean_squared_error(y_test_array[:, i], y_pred_array[:, i])
        rmse = np.sqrt(mse)

        metrics_list.append([back_targets[i], r2, mse, rmse])

        plt.figure(figsize=(6, 5))
        plt.scatter(y_test_array[:, i], y_pred_array[:, i], color='green', alpha=0.6, edgecolors='k', label='Predictions')
        plt.plot([y_test_array[:, i].min(), y_test_array[:, i].max()],
                 [y_test_array[:, i].min(), y_test_array[:, i].max()], 'r--', label='Ideal Fit')

        coef = np.polyfit(y_test_array[:, i], y_pred_array[:, i], 1)
        fit_line = np.poly1d(coef)
        x_fit = np.linspace(y_test_array[:, i].min(), y_test_array[:, i].max(), 100)
        plt.plot(x_fit, fit_line(x_fit), 'b-', label='Regression Fit')

        fit_equations.append([back_targets[i], f'y = {coef[0]:.4f}x + {coef[1]:.4f}'])

        plt.xlabel(f'Actual {back_targets[i]}')
        plt.ylabel(f'Predicted {back_targets[i]}')
        plt.title(f'Actual vs Predicted ({back_targets[i]})')

        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}',
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'D:\\Predict\\back_scatter_{back_targets[i]}.png', dpi=300)
        plt.show()

    metrics_df = pd.DataFrame(metrics_list, columns=["Target", "R2", "MSE", "RMSE"])
    metrics_df.to_excel('D:\\Predict\\model_metrics.xlsx', index=False)
    print("所有指标保存到 D:\\Predict\\model_metrics.xlsx ！")

# 保存拟合公式
fit_eq_df = pd.DataFrame(fit_equations, columns=["Target", "Fit Equation"])
fit_eq_df.to_excel('D:\\Predict\\fit_equations.xlsx', index=False)
print("拟合公式保存到 D:\\Predict\\fit_equations.xlsx ！")
