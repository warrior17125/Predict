# 这一版本最终代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 创建可视化文件夹结构（修正版）
def create_folders():
    base_folders = [
        'metrics_plots',
        'scatter_plots/forward/Forward_Prediction',
        'scatter_plots/reverse',
        'dot_violin_plots/forward',
        'dot_violin_plots/reverse', 
        'decision_plots',
        'waterfall_plots',
        'heatmap_plots',
        'shap_summary_plots'
    ]
    
    for folder in base_folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def safe_savefig(path, dpi=300, bbox_inches='tight'):
    """确保目标目录存在后保存图片"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def train_evaluate_plot(models, X_train, X_test, y_train, y_test, task_type, task_name):
    results = {}
    predictions = {}
    shap_values_dict = {}  # 添加SHAP值存储
    
    for model_name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        results[model_name] = {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        predictions[model_name] = y_pred
        
        # ========== 可视化部分 ==========
        # 散点图（修正路径）
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'{task_name} - {model_name}')
        safe_savefig(f'scatter_plots/{task_type}/{task_name}/{model_name}.png')
        
        # ========== 修正Violin Plot警告 ==========
        plt.figure(figsize=(10,6))
        # 添加hue参数并设置legend
        sns.violinplot(x=y_test, y=y_pred, hue=y_test, 
                       inner="stick", palette="Set2", legend=False)
        # 替换ci为errorbar
        sns.pointplot(x=y_test, y=y_pred, color='black', errorbar=None)
        plt.title(f'Violin Plot - {model_name}')
        safe_savefig(f'dot_violin_plots/{task_name}/{model_name}.png')
        
        # ========== SHAP可视化部分 ==========
        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap_values_dict[model_name] = shap_values
            
            # 决策图
            plt.figure()
            shap.decision_plot(explainer.expected_value, 
                             shap_values.values[0], 
                             X_test.iloc[0], 
                             show=False)
            plt.title(f'Decision Plot - {model_name}')
            safe_savefig(f'decision_plots/{task_name}_{model_name}_decision.png')
            
            # 瀑布图
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title(f'Waterfall Plot - {model_name}')
            safe_savefig(f'waterfall_plots/{task_name}_{model_name}_waterfall.png')
            
            # SHAP概要图
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.title(f'SHAP Summary - {model_name}')
            safe_savefig(f'shap_summary_plots/{task_name}_{model_name}_summary.png')
            
        except Exception as e:
            print(f"SHAP可视化失败: {str(e)}")
            shap_values_dict[model_name] = None
        
    return results, predictions, shap_values_dict  # 返回三个值

# 指标可视化函数
def visualize_metrics(metrics_dict, task_name):
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    # 柱状图
    plt.figure(figsize=(10,6))
    df.plot(kind='bar', rot=0)
    plt.title(f'Performance Metrics - {task_name}')
    plt.ylabel('Score')
    plt.savefig(f'metrics_plots/{task_name}_metrics_bar.png', dpi=300)
    plt.close()
    
    # 热力图
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, cmap='Blues', fmt=".3f")
    plt.title(f'Metrics Heatmap - {task_name}')
    plt.savefig(f'metrics_plots/{task_name}_metrics_heatmap.png', dpi=300)
    plt.close()

# 反向预测处理函数（更新版）
def reverse_predict(models, X_train, X_test, y_train, y_test, task_name):
    metrics = {}
    predictions = {}
    
    for model_name, model in models.items():
        model_metrics = {}
        model_preds = {}
        
        for col in y_train.columns:
            # 训练模型
            model.fit(X_train, y_train[col])
            y_pred = model.predict(X_test)
            
            # 存储结果
            model_metrics[col] = {
                'R2': r2_score(y_test[col], y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test[col], y_pred))
            }
            model_preds[col] = y_pred
            
            # 散点图
            plt.figure(figsize=(8,6))
            plt.scatter(y_test[col], y_pred, alpha=0.6)
            plt.plot([y_test[col].min(), y_test[col].max()], 
                     [y_test[col].min(), y_test[col].max()], 'k--', lw=2)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'{task_name} - {model_name} - {col}')
            plt.savefig(f'scatter_plots/reverse/{model_name}_{col}.png', dpi=300)
            plt.close()
            
            # 残差热力图
            residual = y_pred - y_test[col].values
            plt.figure(figsize=(10,6))
            sns.heatmap(pd.DataFrame({'Pred':y_pred, 'True':y_test[col], 'Residual':residual}).corr(),
                        annot=True, cmap='coolwarm')
            plt.title(f'Correlation Heatmap - {model_name} - {col}')
            plt.savefig(f'heatmap_plots/reverse_{model_name}_{col}.png', dpi=300)
            plt.close()
        
        metrics[model_name] = model_metrics
        predictions[model_name] = model_preds
        
        # 反向预测指标可视化
        metric_df = pd.DataFrame.from_dict(model_metrics, orient='index')
        plt.figure(figsize=(12,6))
        metric_df.plot(kind='bar', rot=45)
        plt.title(f'Reverse Metrics - {model_name}')
        plt.savefig(f'metrics_plots/reverse_{model_name}_metrics.png', dpi=300)
        plt.close()
    
    return metrics, predictions

# 主函数更新
def main(input_path, output_path):
    create_folders()
    df = pd.read_excel(input_path)
    
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
    
    forward_metrics, forward_preds, _ = train_evaluate_plot(
        models, X_train_f, X_test_f, y_train_f, y_test_f, 
        'forward', 'Forward_Prediction')
    
    visualize_metrics(forward_metrics, 'Forward_Prediction')
    
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
        
        # 保存正向预测值
        forward_pred_df = pd.DataFrame({
            'True_Viscosity': y_test_f,
            'RF_Pred': forward_preds['RandomForest'],
            'XGB_Pred': forward_preds['XGBoost'],
            'LGBM_Pred': forward_preds['LightGBM']
        })
        forward_pred_df.to_excel(writer, sheet_name='Forward_Predictions', index=False)
        
        # 保存反向指标
        reverse_metrics_df = pd.concat(
            {k: pd.DataFrame(v) for k, v in reverse_metrics.items()}, 
            axis=1
        ).T
        reverse_metrics_df.to_excel(writer, sheet_name='Reverse_Metrics')
        
        # 保存反向预测值
        for model_name in reverse_preds:
            # 创建基础数据框
            preds_df = pd.DataFrame({
                'Temp': X_test_r['Temp'],
                'Viscosity': X_test_r['Viscosity']
            })
            
            # 添加各特征的预测结果
            for col in reverse_preds[model_name]:
                preds_df[f'{col}_True'] = y_test_r[col].values
                preds_df[f'{col}_Pred'] = reverse_preds[model_name][col]
            
            # 保存到不同sheet
            preds_df.to_excel(
                writer, 
                sheet_name=f'Reverse_{model_name}_Predictions', 
                index=False
            )
        
if __name__ == "__main__":
    input_file = "data.xlsx"
    output_file = "results.xlsx"
    main(input_file, output_file)