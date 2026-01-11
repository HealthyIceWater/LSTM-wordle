import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LSTMPredictor(nn.Module):
    """LSTM预测模型"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # 成功预测的分类头
        self.success_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM输出
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 回归预测（猜测步数）
        regression_output = self.fc(context_vector)

        # 分类预测（是否成功）
        classification_output = self.success_head(context_vector)

        return regression_output, classification_output, attention_weights


class TransformerPredictor(nn.Module):
    """Transformer预测模型"""

    def __init__(self, input_size, hidden_size=64, num_heads=4, num_layers=2, dropout=0.3):
        super(TransformerPredictor, self).__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(input_size, hidden_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输出层
        self.fc_regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.fc_classification = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 嵌入和位置编码
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)

        # Transformer编码
        transformer_out = self.transformer(embedded)

        # 取最后一个时间步
        last_output = transformer_out[:, -1, :]

        # 预测
        regression_output = self.fc_regression(last_output)
        classification_output = self.fc_classification(last_output)

        return regression_output, classification_output


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(model_type, input_size, **kwargs):
        """创建模型"""
        if model_type == 'lstm':
            return LSTMPredictor(input_size, **kwargs)
        elif model_type == 'transformer':
            return TransformerPredictor(input_size, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")