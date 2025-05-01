# 只有预测结果散点图可视化 + 预测结果&评价指标保存
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 读取数据
def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

# 模型训练与评估函数
def train_evaluate_plot(models, X_train, X_test, y_train, y_test, task_name):
    results = {}
    predictions = {}
    for model_name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        # 预测
        y_pred = model.predict(X_test)
        # 计算指标
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[model_name] = {'R2': r2, 'RMSE': rmse}
        predictions[model_name] = y_pred
        
        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predictions', fontsize=12)
        plt.title(f'{task_name} - {model_name}', fontsize=14)
        plt.grid(True)
        plt.savefig(f'{task_name}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results, predictions

# 反向预测处理函数
def reverse_predict(models, X_train, X_test, y_train, y_test, task_name):
    metrics = {}
    predictions = {}
    for model_name, model in models.items():
        model_metrics = {}
        model_preds = {}
        for col in y_train.columns:
            # 训练模型
            model.fit(X_train, y_train[col])
            # 预测
            y_pred = model.predict(X_test)
            # 计算指标
            r2 = r2_score(y_test[col], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test[col], y_pred))
            model_metrics[col] = {'R2': r2, 'RMSE': rmse}
            model_preds[col] = y_pred
            
            # 绘制散点图
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test[col], y_pred, alpha=0.6)
            plt.plot([y_test[col].min(), y_test[col].max()], 
                     [y_test[col].min(), y_test[col].max()], 'k--', lw=2)
            plt.xlabel('True Values', fontsize=12)
            plt.ylabel('Predictions', fontsize=12)
            plt.title(f'{task_name} - {model_name} - {col}', fontsize=14)
            plt.grid(True)
            plt.savefig(f'Reverse_{model_name}_{col}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        metrics[model_name] = model_metrics
        predictions[model_name] = model_preds
    
    return metrics, predictions

# 主函数
def main(input_path, output_path):
    # 加载数据
    df = load_data(input_path)
    
    # 定义模型
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    # 正向预测
    X_forward = df.drop(columns=['Viscosity'])
    y_forward = df['Viscosity']
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_forward, y_forward, test_size=0.2, random_state=42)
    
    forward_metrics, forward_preds = train_evaluate_plot(
        models, X_train_f, X_test_f, y_train_f, y_test_f, 'Forward_Prediction')
    
    # 反向预测
    X_reverse = df[['Temp', 'Viscosity']]
    y_reverse = df.drop(columns=['Temp', 'Viscosity'])  # 前11列
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reverse, y_reverse, test_size=0.2, random_state=42)
    
    reverse_metrics, reverse_preds = reverse_predict(
        models, X_train_r, X_test_r, y_train_r, y_test_r, 'Reverse_Prediction')
    
    # 保存结果到Excel
    with pd.ExcelWriter(output_path) as writer:
        # 保存正向结果
        pd.DataFrame.from_dict(forward_metrics, orient='index').to_excel(
            writer, sheet_name='Forward_Metrics')
        pd.DataFrame({
            'True_Viscosity': y_test_f,
            'RF_Pred': forward_preds['RandomForest'],
            'XGB_Pred': forward_preds['XGBoost'],
            'LGBM_Pred': forward_preds['LightGBM']
        }).to_excel(writer, sheet_name='Forward_Predictions', index=False)
        
        # 保存反向结果
        for model in reverse_metrics:
            metrics_df = pd.DataFrame.from_dict(reverse_metrics[model], orient='index')
            metrics_df.to_excel(writer, sheet_name=f'Reverse_{model}_Metrics')
            
            preds_df = pd.DataFrame({
                'Temp': X_test_r['Temp'],
                'Viscosity': X_test_r['Viscosity']
            })
            for col in reverse_preds[model]:
                preds_df[f'{col}_True'] = y_test_r[col].values
                preds_df[f'{col}_Pred'] = reverse_preds[model][col]
            preds_df.to_excel(writer, sheet_name=f'Reverse_{model}_Predictions', index=False)

if __name__ == "__main__":
    input_file = "data.xlsx"    # 修改为实际输入路径
    output_file = "res1.xlsx"     # 修改为实际输出路径
    main(input_file, output_file)