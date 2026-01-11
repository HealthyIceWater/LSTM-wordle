import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer预测模型"""

    def __init__(self, input_size, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 回归头（预测猜测次数）
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # 分类头（预测是否成功）
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.d_model = d_model

    def forward(self, src, return_attention=False):
        # 输入投影
        src = self.input_projection(src) * np.sqrt(self.d_model)

        # 位置编码
        src = self.pos_encoder(src)

        # Transformer编码器
        transformer_output = self.transformer_encoder(src)

        # 使用最后一个时间步的输出
        last_output = transformer_output[:, -1, :]

        # 回归预测
        attempts_pred = self.regression_head(last_output)

        # 分类预测
        success_pred = self.classification_head(last_output)

        return attempts_pred, success_pred


def prepare_transformer_data(df, sequence_length=10):
    """为Transformer准备数据（与LSTM相同）"""
    sequences = []
    targets_attempts = []
    targets_success = []

    for player_id in df['player_id'].unique():
        player_data = df[df['player_id'] == player_id].sort_values('game_number')

        feature_cols = ['avg_attempts_last5', 'std_attempts_last5',
                        'success_rate_last5', 'avg_time_last5', 'avg_interval_last5']

        player_features = player_data[feature_cols].values

        for i in range(sequence_length, len(player_features)):
            sequence = player_features[i - sequence_length:i]
            target_attempt = player_data.iloc[i]['target_attempts']
            target_success = player_data.iloc[i]['target_success']

            sequences.append(sequence)
            targets_attempts.append(target_attempt)
            targets_success.append(target_success)

    return np.array(sequences), np.array(targets_attempts), np.array(targets_success)


def train_transformer(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0005):
    """训练Transformer模型"""
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch_X, (batch_y_attempts, batch_y_success) in train_loader:
            optimizer.zero_grad()

            attempts_pred, success_pred = model(batch_X)

            loss_attempts = regression_criterion(attempts_pred.squeeze(), batch_y_attempts)
            loss_success = classification_criterion(success_pred.squeeze(), batch_y_success)
            loss = loss_attempts + loss_success

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, (batch_y_attempts, batch_y_success) in val_loader:
                attempts_pred, success_pred = model(batch_X)

                loss_attempts = regression_criterion(attempts_pred.squeeze(), batch_y_attempts)
                loss_success = classification_criterion(success_pred.squeeze(), batch_y_success)
                loss = loss_attempts + loss_success

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

    return train_losses, val_losses


def evaluate_transformer(model, test_loader):
    """评估Transformer模型"""
    model.eval()
    all_attempts_pred = []
    all_attempts_true = []
    all_success_pred = []
    all_success_true = []

    with torch.no_grad():
        for batch_X, (batch_y_attempts, batch_y_success) in test_loader:
            attempts_pred, success_pred = model(batch_X)

            all_attempts_pred.extend(attempts_pred.squeeze().numpy())
            all_attempts_true.extend(batch_y_attempts.numpy())
            all_success_pred.extend(success_pred.squeeze().numpy())
            all_success_true.extend(batch_y_success.numpy())

    # 计算指标
    attempts_pred = np.array(all_attempts_pred)
    attempts_true = np.array(all_attempts_true)

    mae = mean_absolute_error(attempts_true, attempts_pred)
    rmse = np.sqrt(mean_squared_error(attempts_true, attempts_pred))

    success_pred = np.array(all_success_pred)
    success_true = np.array(all_success_true)
    success_pred_class = (success_pred > 0.5).astype(int)

    f1 = f1_score(success_true, success_pred_class)
    auc = roc_auc_score(success_true, success_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'F1': f1,
        'AUC': auc,
        'attempts_pred': attempts_pred,
        'attempts_true': attempts_true,
        'success_pred': success_pred,
        'success_true': success_true
    }


def compare_models(lstm_results, transformer_results):
    """比较两个模型的性能"""
    comparison = pd.DataFrame({
        'Model': ['LSTM', 'Transformer'],
        'MAE': [lstm_results['MAE'], transformer_results['MAE']],
        'RMSE': [lstm_results['RMSE'], transformer_results['RMSE']],
        'F1': [lstm_results['F1'], transformer_results['F1']],
        'AUC': [lstm_results['AUC'], transformer_results['AUC']]
    })

    return comparison


def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_excel('wordle_aggregated_data.xlsx')

    # 准备数据
    print("准备Transformer数据...")
    X, y_attempts, y_success = prepare_transformer_data(df, sequence_length=10)

    # 划分数据集
    X_train, X_temp, y_attempts_train, y_attempts_temp, y_success_train, y_success_temp = train_test_split(
        X, y_attempts, y_success, test_size=0.3, random_state=42
    )

    X_val, X_test, y_attempts_val, y_attempts_test, y_success_val, y_success_test = train_test_split(
        X_temp, y_attempts_temp, y_success_temp, test_size=0.5, random_state=42
    )

    # 创建数据加载器
    from lstm_model import WordleDataset  # 复用相同的Dataset类

    train_dataset = WordleDataset(X_train, y_attempts_train, y_success_train)
    val_dataset = WordleDataset(X_val, y_attempts_val, y_success_val)
    test_dataset = WordleDataset(X_test, y_attempts_test, y_success_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建Transformer模型
    input_size = X.shape[2]
    model = TransformerModel(input_size=input_size, d_model=64, nhead=4, num_layers=3)

    print(f"Transformer参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 训练模型
    print("开始训练Transformer模型...")
    train_losses, val_losses = train_transformer(
        model, train_loader, val_loader,
        num_epochs=50, learning_rate=0.0005
    )

    # 评估模型
    print("\n评估Transformer模型性能...")
    results = evaluate_transformer(model, test_loader)

    print(f"MAE: {results['MAE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")
    print(f"F1 Score: {results['F1']:.4f}")
    print(f"AUC: {results['AUC']:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer_wordle_model.pth')
    print("Transformer模型已保存到 transformer_wordle_model.pth")

    # 保存结果
    results_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'F1', 'AUC'],
        'Value': [results['MAE'], results['RMSE'], results['F1'], results['AUC']]
    })
    results_df.to_excel('transformer_results.xlsx', index=False)

    # 比较两个模型（需要LSTM结果）
    try:
        lstm_results_df = pd.read_excel('lstm_results.xlsx')
        lstm_results = {
            'MAE': lstm_results_df.loc[0, 'Value'],
            'RMSE': lstm_results_df.loc[1, 'Value'],
            'F1': lstm_results_df.loc[2, 'Value'],
            'AUC': lstm_results_df.loc[3, 'Value']
        }

        comparison = compare_models(lstm_results, results)
        print("\n模型比较:")
        print(comparison)
        comparison.to_excel('model_comparison.xlsx', index=False)
    except:
        print("未找到LSTM结果，跳过比较")

    return model, results, train_losses, val_losses


if __name__ == "__main__":
    model, results, train_losses, val_losses = main()