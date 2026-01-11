import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """模型训练器"""

    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model, train_loader, test_loader, epochs=100, lr=0.001, model_name='model'):
        """训练模型"""
        model = model.to(self.device)

        # 损失函数和优化器
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        print(f"开始训练{model_name}模型...")
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                # 分离标签
                batch_y_attempts, batch_y_success = batch_y
                batch_X = batch_X.to(self.device)
                batch_y_attempts = batch_y_attempts.to(self.device)
                batch_y_success = batch_y_success.to(self.device)

                optimizer.zero_grad()

                # 获取模型输出
                outputs = model(batch_X)

                # 根据模型类型处理输出
                if len(outputs) == 3:
                    pred_attempts, pred_success, _ = outputs
                else:
                    pred_attempts, pred_success = outputs

                # 多任务损失
                loss1 = mse_loss(pred_attempts, batch_y_attempts)
                loss2 = bce_loss(pred_success, batch_y_success)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    # 分离标签
                    batch_y_attempts, batch_y_success = batch_y
                    batch_X = batch_X.to(self.device)
                    batch_y_attempts = batch_y_attempts.to(self.device)
                    batch_y_success = batch_y_success.to(self.device)

                    outputs = model(batch_X)

                    if len(outputs) == 3:
                        pred_attempts, pred_success, _ = outputs
                    else:
                        pred_attempts, pred_success = outputs

                    loss1 = mse_loss(pred_attempts, batch_y_attempts)
                    loss2 = bce_loss(pred_success, batch_y_success)
                    loss = loss1 + loss2

                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'best_{model_name}.pth')

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        return train_losses, val_losses

    def evaluate(self, model, test_loader, y_test_attempts, y_test_success, model_name='model'):
        """评估模型性能"""
        # 加载最佳模型
        model_path = f'best_{model_name}.pth'
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"警告: 未找到模型文件 {model_path}，使用当前模型进行评估")

        model.eval()

        all_pred_attempts = []
        all_pred_success = []
        all_true_attempts = []
        all_true_success = []
        attention_weights_list = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # 分离标签
                batch_y_attempts, batch_y_success = batch_y
                batch_X = batch_X.to(self.device)

                outputs = model(batch_X)

                if len(outputs) == 3:
                    pred_attempts, pred_success, attention_weights = outputs
                    # 保存注意力权重
                    if attention_weights is not None:
                        attention_weights_list.append(attention_weights.cpu().numpy())
                else:
                    pred_attempts, pred_success = outputs

                all_pred_attempts.extend(pred_attempts.cpu().numpy())
                all_pred_success.extend(pred_success.cpu().numpy())
                all_true_attempts.extend(batch_y_attempts.numpy())
                all_true_success.extend(batch_y_success.numpy())

        # 转换为numpy数组
        all_pred_attempts = np.array(all_pred_attempts).flatten()
        all_pred_success = np.array(all_pred_success).flatten()
        all_true_attempts = np.array(all_true_attempts).flatten()
        all_true_success = np.array(all_true_success).flatten()

        # 回归任务评估
        mae = mean_absolute_error(all_true_attempts, all_pred_attempts)
        rmse = np.sqrt(mean_squared_error(all_true_attempts, all_pred_attempts))

        # 分类任务评估
        pred_success_binary = (all_pred_success > 0.5).astype(int)
        f1 = f1_score(all_true_success, pred_success_binary)
        auc = roc_auc_score(all_true_success, all_pred_success)

        print(f"\n{model_name.upper()}模型性能评估结果:")
        print(f"回归任务 (预测猜测步数):")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"分类任务 (预测是否成功):")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        results = {
            'mae': mae,
            'rmse': rmse,
            'f1': f1,
            'auc': auc,
            'predictions': {
                'attempts': (all_true_attempts, all_pred_attempts),
                'success': (all_true_success, all_pred_success)
            }
        }

        if attention_weights_list:
            results['attention_weights'] = np.concatenate(attention_weights_list, axis=0)

        return results


class Visualizer:
    """可视化类"""

    @staticmethod
    def plot_training_history(train_losses, val_losses, model_name='Model'):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label=f'{model_name} Train Loss', color='blue', linewidth=2)
        plt.plot(val_losses, label=f'{model_name} Val Loss', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_comprehensive_results(results_dict, feature_names=None):
        """绘制综合结果"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        model_names = list(results_dict.keys())

        # 1. 损失曲线对比
        for i, model_name in enumerate(model_names):
            train_losses = results_dict[model_name]['train_losses']
            val_losses = results_dict[model_name]['val_losses']
            axes[0, 0].plot(train_losses, label=f'{model_name} Train', alpha=0.7)
            axes[0, 0].plot(val_losses, label=f'{model_name} Val', linestyle='--', alpha=0.7)

        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('训练和验证损失曲线对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 预测结果趋势图
        for i, model_name in enumerate(model_names[:1]):  # 只显示第一个模型
            eval_results = results_dict[model_name]['evaluation']
            true_attempts, pred_attempts = eval_results['predictions']['attempts']
            axes[0, 1].plot(true_attempts[:50], label='真实值', marker='o', alpha=0.7)
            axes[0, 1].plot(pred_attempts[:50], label='预测值', marker='s', alpha=0.7)

        axes[0, 1].set_xlabel('样本序号')
        axes[0, 1].set_ylabel('平均猜测步数')
        axes[0, 1].set_title('猜测步数预测结果趋势')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ROC曲线对比
        for i, model_name in enumerate(model_names):
            eval_results = results_dict[model_name]['evaluation']
            true_success, pred_success = eval_results['predictions']['success']
            fpr, tpr, _ = roc_curve(true_success, pred_success)
            axes[1, 0].plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {eval_results["auc"]:.3f})')

        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlabel('假正率')
        axes[1, 0].set_ylabel('真正率')
        axes[1, 0].set_title('ROC曲线对比')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 性能指标对比
        metrics = ['MAE', 'RMSE', 'F1-Score', 'AUC']
        metric_data = []
        for i, metric in enumerate(metrics):
            row, col = 1 + i // 2, i % 2
            values = []
            for model_name in model_names:
                eval_results = results_dict[model_name]['evaluation']
                if metric == 'MAE':
                    values.append(eval_results['mae'])
                elif metric == 'RMSE':
                    values.append(eval_results['rmse'])
                elif metric == 'F1-Score':
                    values.append(eval_results['f1'])
                elif metric == 'AUC':
                    values.append(eval_results['auc'])

            # 确保有足够的子图
            if row < 3 and col < 2:
                bars = axes[row, col].bar(model_names, values)
                axes[row, col].set_title(f'{metric}对比')
                axes[row, col].set_ylabel(metric)
                axes[row, col].grid(True, alpha=0.3)

                for bar, value in zip(bars, values):
                    axes[row, col].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                        f'{value:.3f}', ha='center', va='bottom')
            metric_data.append((metric, values))

        # 5. 混淆矩阵（只显示第一个模型）
        if model_names:
            model_name = model_names[0]
            eval_results = results_dict[model_name]['evaluation']
            true_success, pred_success = eval_results['predictions']['success']
            pred_success_binary = (pred_success > 0.5).astype(int)
            cm = confusion_matrix(true_success, pred_success_binary)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
            axes[2, 0].set_xlabel('预测标签')
            axes[2, 0].set_ylabel('真实标签')
            axes[2, 0].set_title(f'{model_name}混淆矩阵')

        # 6. 性能指标汇总表格
        if metric_data and len(axes) > 5:
            axes[2, 1].axis('tight')
            axes[2, 1].axis('off')

            # 创建表格数据
            table_data = []
            headers = ['Metric'] + model_names

            for metric, values in metric_data:
                row = [metric] + [f'{v:.3f}' for v in values]
                table_data.append(row)

            table = axes[2, 1].table(cellText=table_data, colLabels=headers,
                                     loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[2, 1].set_title('性能指标汇总')

        plt.suptitle('模型性能综合对比', fontsize=16)
        plt.tight_layout()
        plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_attention_weights(attention_weights, feature_names):
        """绘制注意力权重热力图"""
        if attention_weights is not None and len(attention_weights) > 0:
            plt.figure(figsize=(12, 8))

            # 取第一个样本的注意力权重
            attention_sample = attention_weights[0].squeeze()

            if len(attention_sample.shape) == 2:
                sns.heatmap(attention_sample.T, cmap='viridis',
                            xticklabels=[f'Step {i + 1}' for i in range(attention_sample.shape[0])],
                            yticklabels=feature_names if feature_names else None)
                plt.xlabel('时间步')
                plt.ylabel('特征维度')
                plt.title('注意力权重热力图（第一个样本）')
                plt.tight_layout()
                plt.savefig('attention_heatmap.png', dpi=300, bbox_inches='tight')
                plt.show()

    @staticmethod
    def plot_predictions_comparison(true_values, pred_values, model_name='Model', metric_name='Attempts'):
        """绘制预测值对比图"""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(true_values, pred_values, alpha=0.5)
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()],
                 'r--', label='理想预测线')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{model_name} - {metric_name}预测散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        residuals = true_values - pred_values
        plt.scatter(pred_values, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title(f'{model_name} - {metric_name}残差图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_{metric_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_feature_importance(attention_weights, feature_names):
        """绘制特征重要性图"""
        if attention_weights is not None and len(attention_weights) > 0:
            # 计算平均注意力权重
            avg_attention = np.mean(attention_weights, axis=(0, 1))  # 平均批次和时间步

            if len(avg_attention) == len(feature_names):
                # 按重要性排序
                indices = np.argsort(avg_attention)[::-1]
                sorted_features = [feature_names[i] for i in indices]
                sorted_importance = avg_attention[indices]

                plt.figure(figsize=(12, 8))
                bars = plt.barh(range(len(sorted_features)), sorted_importance)
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel('平均注意力权重')
                plt.title('特征重要性（基于注意力机制）')
                plt.grid(True, alpha=0.3, axis='x')

                # 为条形图添加数值标签
                for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
                    plt.text(value + 0.001, bar.get_y() + bar.get_height() / 2,
                             f'{value:.3f}', va='center')

                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()

                # 打印特征重要性排名
                print("\n特征重要性排名:")
                for i, idx in enumerate(indices):
                    print(f"  {i + 1}. {feature_names[idx]}: {avg_attention[idx]:.4f}")