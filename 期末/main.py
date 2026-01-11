# improved_main.py
import torch
import numpy as np
import pandas as pd
import sys
import os
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from models import ModelFactory
from trainer import ModelTrainer, Visualizer
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class ImprovedModelTrainer(ModelTrainer):
    """改进的模型训练器"""

    def __init__(self, device=None):
        super().__init__(device)

    def train_improved(self, model, train_loader, test_loader, epochs=100, lr=0.001, model_name='model',
                       regression_weight=1.0, classification_weight=1.0):
        """改进的训练方法"""
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
            train_regression_loss = 0
            train_classification_loss = 0

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

                # 计算两个任务的损失
                loss_regression = mse_loss(pred_attempts, batch_y_attempts)
                loss_classification = bce_loss(pred_success, batch_y_success)

                # 加权组合损失
                loss = regression_weight * loss_regression + classification_weight * loss_classification

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_regression_loss += loss_regression.item()
                train_classification_loss += loss_classification.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            val_regression_loss = 0
            val_classification_loss = 0

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

                    loss_regression = mse_loss(pred_attempts, batch_y_attempts)
                    loss_classification = bce_loss(pred_success, batch_y_success)
                    loss = regression_weight * loss_regression + classification_weight * loss_classification

                    val_loss += loss.item()
                    val_regression_loss += loss_regression.item()
                    val_classification_loss += loss_classification.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            avg_train_regression = train_regression_loss / len(train_loader)
            avg_train_classification = train_classification_loss / len(train_loader)
            avg_val_regression = val_regression_loss / len(test_loader)
            avg_val_classification = val_classification_loss / len(test_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'best_{model_name}.pth')

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}]')
                print(
                    f'  Train: Total={avg_train_loss:.4f}, Reg={avg_train_regression:.4f}, Cls={avg_train_classification:.4f}')
                print(
                    f'  Val:   Total={avg_val_loss:.4f}, Reg={avg_val_regression:.4f}, Cls={avg_val_classification:.4f}')

        return train_losses, val_losses

    def evaluate_improved(self, model, test_loader, y_test_attempts, y_test_success, model_name='model'):
        """改进的评估方法"""
        try:
            model.load_state_dict(torch.load(f'best_{model_name}.pth', map_location=self.device))
        except FileNotFoundError:
            print(f"警告: 未找到模型文件 best_{model_name}.pth")

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
        mae_attempts = mean_absolute_error(all_true_attempts, all_pred_attempts)
        rmse_attempts = np.sqrt(mean_squared_error(all_true_attempts, all_pred_attempts))

        # 分类任务评估
        pred_success_binary = (all_pred_success > 0.5).astype(int)
        true_success_binary = (all_true_success > 0.5).astype(int)

        # 检查是否有正样本
        positive_samples = np.sum(true_success_binary)
        total_samples = len(true_success_binary)

        if positive_samples == 0:
            print(f"警告: 测试集中没有正样本（成功样本）")
            f1 = 0
            auc = 0
        elif positive_samples == total_samples:
            print(f"警告: 测试集中所有样本都是正样本")
            f1 = 1.0 if np.all(pred_success_binary == 1) else 0
            auc = 1.0 if np.all(pred_success_binary == 1) else 0
        else:
            f1 = f1_score(true_success_binary, pred_success_binary)
            try:
                auc = roc_auc_score(true_success_binary, all_pred_success)
            except:
                auc = 0.5  # 如果AUC计算失败，设为0.5（随机猜测）

        # 打印分类报告
        print(f"\n分类任务详细报告:")
        print(f"正样本数: {positive_samples}/{total_samples} ({positive_samples / total_samples * 100:.1f}%)")
        print(classification_report(true_success_binary, pred_success_binary,
                                    target_names=['失败', '成功'], zero_division=0))

        results = {
            'mae_attempts': mae_attempts,
            'rmse_attempts': rmse_attempts,
            'f1': f1,
            'auc': auc,
            'positive_ratio': positive_samples / total_samples,
            'predictions': {
                'attempts': (all_true_attempts, all_pred_attempts),
                'success': (all_true_success, all_pred_success),
                'success_binary': (true_success_binary, pred_success_binary)
            }
        }

        if attention_weights_list:
            results['attention_weights'] = np.concatenate(attention_weights_list, axis=0)

        return results


def analyze_data_distribution(filepath):
    """分析数据分布"""
    print("分析数据分布...")

    # 读取数据
    df = pd.read_excel(filepath)

    # 计算平均猜测步数
    def calculate_avg_attempts(row):
        total = row['Number of reported results']
        attempts = 0
        weights = [1, 2, 3, 4, 5, 6, 7]

        for i, w in enumerate(weights, 1):
            col_name = f"{i} tries" if i <= 6 else "7 or more tries (X)"
            attempts += row[col_name] * w

        return attempts / total

    df['avg_attempts'] = df.apply(calculate_avg_attempts, axis=1)

    # 计算成功率
    def calculate_success_rate(row):
        total = row['Number of reported results']
        success = sum(row[f"{i} tries"] for i in range(1, 7))
        return success / total

    df['success_rate'] = df.apply(calculate_success_rate, axis=1)

    # 分析数据
    print(f"\n猜测步数分析:")
    print(f"  均值: {df['avg_attempts'].mean():.3f}")
    print(f"  标准差: {df['avg_attempts'].std():.3f}")
    print(f"  范围: [{df['avg_attempts'].min():.3f}, {df['avg_attempts'].max():.3f}]")

    print(f"\n成功率分析:")
    print(f"  均值: {df['success_rate'].mean():.3f}")
    print(f"  标准差: {df['success_rate'].std():.3f}")
    print(f"  范围: [{df['success_rate'].min():.3f}, {df['success_rate'].max():.3f}]")

    # 分析成功率阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n不同阈值下的正样本比例:")
    for threshold in thresholds:
        positive_ratio = (df['success_rate'] > threshold).mean()
        print(f"  阈值={threshold}: {positive_ratio:.1%}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 猜测步数分布
    axes[0, 0].hist(df['avg_attempts'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['avg_attempts'].mean(), color='red', linestyle='--',
                       label=f'均值={df["avg_attempts"].mean():.2f}')
    axes[0, 0].set_xlabel('平均猜测步数')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('猜测步数分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 成功率分布
    axes[0, 1].hist(df['success_rate'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['success_rate'].mean(), color='red', linestyle='--',
                       label=f'均值={df["success_rate"].mean():.2f}')
    for threshold in thresholds:
        axes[0, 1].axvline(threshold, color='green', linestyle=':', alpha=0.5, label=f'阈值={threshold}')
    axes[0, 1].set_xlabel('成功率')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].set_title('成功率分布')
    axes[0, 1].legend(fontsize='small')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 时间序列图
    axes[1, 0].plot(df['avg_attempts'].values, label='猜测步数', alpha=0.7)
    axes[1, 0].plot(df['success_rate'].values * 7, label='成功率×7', alpha=0.7)  # 缩放以便在同一图中显示
    axes[1, 0].set_xlabel('天数')
    axes[1, 0].set_ylabel('值')
    axes[1, 0].set_title('时间序列')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 散点图：猜测步数 vs 成功率
    axes[1, 1].scatter(df['avg_attempts'], df['success_rate'], alpha=0.5)
    axes[1, 1].set_xlabel('平均猜测步数')
    axes[1, 1].set_ylabel('成功率')
    axes[1, 1].set_title('猜测步数 vs 成功率')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 建议最佳阈值
    median_success = df['success_rate'].median()
    mean_success = df['success_rate'].mean()

    print(f"\n建议:")
    print(f"  成功率中位数: {median_success:.3f}")
    print(f"  成功率平均值: {mean_success:.3f}")
    print(f"  建议分类阈值: {mean_success:.3f} (平均值)")

    return df, mean_success


def adjust_threshold_in_processor(processor, threshold):
    """调整数据处理器的成功率阈值"""
    print(f"调整分类阈值: {threshold}")

    # 修改_create_sequences方法中的阈值
    import types

    def new_create_sequences(self, df, success_threshold=None):
        """新的创建序列方法，使用可调节的阈值"""
        if success_threshold is None:
            success_threshold = 0.5  # 默认值

        features = []
        labels_attempts = []
        labels_success = []

        n_samples = len(df) - self.sequence_length
        print(f"创建时间序列，使用阈值={success_threshold:.3f}，共有{n_samples}个样本")

        for i in range(n_samples):
            # 提取特征
            seq_features = []
            for j in range(self.sequence_length):
                idx = i + j
                feature_vector = []

                # 各种尝试次数的百分比
                for k in range(1, 7):
                    col_name = f"{k} tries"
                    feature_vector.append(df.iloc[idx][col_name] / df.iloc[idx]['Number of reported results'])

                # 添加其他特征
                feature_vector.append(df.iloc[idx]['hard_mode_ratio'])
                feature_vector.append(df.iloc[idx]['Contest number'] / 1000)  # 归一化

                seq_features.append(feature_vector)

            features.append(seq_features)

            # 标签：下一轮的平均猜测步数和成功率
            next_idx = i + self.sequence_length
            labels_attempts.append(df.iloc[next_idx]['avg_attempts'])
            labels_success.append(1 if df.iloc[next_idx]['success_rate'] > success_threshold else 0)

        features = np.array(features)
        labels_attempts = np.array(labels_attempts)
        labels_success = np.array(labels_success)

        print(f"创建的特征形状: {features.shape}")
        print(f"标签数量: {len(labels_attempts)}")
        print(f"正样本比例: {labels_success.mean():.1%}")

        # 数据标准化
        original_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.fit_transform(features_2d)
        features = features_scaled.reshape(original_shape)

        return features, labels_attempts, labels_success

    # 替换方法
    processor._create_sequences = types.MethodType(new_create_sequences, processor)
    return processor


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 分析数据分布
    print("=" * 60)
    print("Wordle玩家表现预测系统 - 改进版")
    print("=" * 60)

    data_file = 'wordle_data.xlsx'
    if not os.path.exists(data_file):
        print(f"数据文件 {data_file} 不存在，请先运行 data_generator.py")
        return

    df, optimal_threshold = analyze_data_distribution(data_file)

    # 2. 使用最优阈值处理数据
    data_processor = DataProcessor(sequence_length=10)
    data_processor = adjust_threshold_in_processor(data_processor, optimal_threshold)

    features, labels_attempts, labels_success = data_processor.load_and_preprocess(data_file)

    print(f"数据加载完成！特征形状: {features.shape}")
    print(f"正样本比例: {labels_success.mean():.1%}")

    # 3. 准备数据
    train_loader, test_loader, y_test_attempts, y_test_success = data_processor.prepare_dataloaders(
        features, labels_attempts, labels_success, batch_size=16, test_size=0.2
    )

    # 4. 定义要训练的模型
    input_size = features.shape[-1]
    print(f"输入特征维度: {input_size}")

    models_to_train = [
        {'type': 'lstm', 'name': 'LSTM', 'params': {'hidden_size': 32, 'num_layers': 2, 'dropout': 0.3}},
        {'type': 'transformer', 'name': 'Transformer',
         'params': {'hidden_size': 32, 'num_heads': 2, 'num_layers': 2, 'dropout': 0.3}}
    ]

    results = {}
    trainer = ImprovedModelTrainer()
    visualizer = Visualizer()

    # 5. 训练和评估每个模型
    for model_config in models_to_train:
        print(f"\n{'-' * 60}")
        print(f"训练{model_config['name']}模型")
        print(f"{'-' * 60}")

        # 创建模型
        model = ModelFactory.create_model(
            model_config['type'],
            input_size,
            **model_config['params']
        )

        print(f"{model_config['name']}模型参数数量: {sum(p.numel() for p in model.parameters())}")

        # 根据正样本比例调整损失权重
        positive_ratio = labels_success.mean()
        if positive_ratio < 0.2 or positive_ratio > 0.8:
            # 样本不平衡，调整分类任务权重
            classification_weight = 2.0
            print(f"样本不平衡，分类任务权重调整为: {classification_weight}")
        else:
            classification_weight = 1.0

        # 训练模型
        train_losses, val_losses = trainer.train_improved(
            model, train_loader, test_loader,
            epochs=50, lr=0.001,
            model_name=model_config['name'].lower(),
            regression_weight=1.0,
            classification_weight=classification_weight
        )

        # 评估模型
        evaluation_results = trainer.evaluate_improved(
            model, test_loader, y_test_attempts, y_test_success,
            model_name=model_config['name'].lower()
        )

        # 保存结果
        results[model_config['name']] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'evaluation': evaluation_results,
            'params': model_config['params']
        }

        # 绘制训练历史
        visualizer.plot_training_history(train_losses, val_losses, model_config['name'])

        # 绘制预测结果对比
        true_attempts, pred_attempts = evaluation_results['predictions']['attempts']
        visualizer.plot_predictions_comparison(
            true_attempts, pred_attempts,
            model_config['name'], 'Attempts'
        )

    # 6. 绘制综合对比结果
    feature_names = ['1次尝试%', '2次尝试%', '3次尝试%', '4次尝试%', '5次尝试%',
                     '6次尝试%', '困难模式比例', '比赛编号']

    visualizer.plot_comprehensive_results(results)

    # 7. 如果有注意力权重，绘制热力图
    if 'LSTM' in results and 'attention_weights' in results['LSTM']['evaluation']:
        visualizer.plot_attention_weights(
            results['LSTM']['evaluation']['attention_weights'],
            feature_names
        )

        # 绘制特征重要性
        visualizer.plot_feature_importance(
            results['LSTM']['evaluation']['attention_weights'],
            feature_names
        )

    # 8. 保存结果到文件
    save_results_to_file(results, features.shape, optimal_threshold)

    # 9. 打印模型对比
    print_comparison_table(results)


def save_results_to_file(results, data_shape, threshold):
    """保存结果到文件"""
    summary = {
        'Data_Info': {
            'Total_Samples': data_shape[0],
            'Sequence_Length': data_shape[1],
            'Feature_Dim': data_shape[2],
            'Optimal_Threshold': threshold
        },
        'Model_Results': {}
    }

    for model_name, model_results in results.items():
        eval_results = model_results['evaluation']
        summary['Model_Results'][model_name] = {
            'Performance': {
                'MAE_Attempts': float(eval_results['mae_attempts']),
                'RMSE_Attempts': float(eval_results['rmse_attempts']),
                'F1_Score': float(eval_results['f1']),
                'AUC': float(eval_results['auc']),
                'Positive_Ratio': float(eval_results['positive_ratio'])
            },
            'Training_Params': model_results['params']
        }

    with open('improved_results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到 'improved_results_summary.json'")


def print_comparison_table(results):
    """打印模型对比表"""
    print("\n" + "=" * 70)
    print("模型性能对比表")
    print("=" * 70)

    headers = ["模型", "MAE(步数)", "RMSE(步数)", "F1-Score", "AUC", "正样本比例"]
    rows = []

    for model_name, model_results in results.items():
        eval_results = model_results['evaluation']
        rows.append([
            model_name,
            f"{eval_results['mae_attempts']:.4f}",
            f"{eval_results['rmse_attempts']:.4f}",
            f"{eval_results['f1']:.4f}",
            f"{eval_results['auc']:.4f}",
            f"{eval_results['positive_ratio']:.1%}"
        ])

    # 创建DataFrame以便更好地显示
    df = pd.DataFrame(rows, columns=headers)
    print(df.to_string(index=False))

    # 找出最佳模型
    if results:
        best_mae = min(results.keys(), key=lambda x: results[x]['evaluation']['mae_attempts'])
        best_f1 = max(results.keys(), key=lambda x: results[x]['evaluation']['f1'])
        best_auc = max(results.keys(), key=lambda x: results[x]['evaluation']['auc'])

        print(f"\n总结:")
        print(f"- 在猜测步数预测上（MAE最低）: {best_mae}")
        print(f"- 在成功率预测上（F1-Score最高）: {best_f1}")
        print(f"- 在分类性能上（AUC最高）: {best_auc}")


if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score

    main()