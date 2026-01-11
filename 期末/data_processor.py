import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class WordleDataset(Dataset):
    """Wordle数据集类"""

    def __init__(self, sequences, labels_attempts, labels_success, sequence_length=10):
        self.sequences = sequences
        self.labels_attempts = labels_attempts
        self.labels_success = labels_success
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label_attempt = torch.FloatTensor([self.labels_attempts[idx]])
        label_success = torch.FloatTensor([self.labels_success[idx]])
        return torch.FloatTensor(sequence), (label_attempt, label_success)


class DataProcessor:
    """数据处理类"""

    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def load_and_preprocess(self, filepath):
        """加载和预处理数据"""
        print("正在加载数据...")

        # 根据文件扩展名选择读取方式
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            # 读取Excel文件，跳过第0行，使用第1行作为表头
            df = pd.read_excel(filepath, header=1)

            # 清理列名
            df.columns = [str(col).strip() for col in df.columns]

            print("数据列名:", df.columns.tolist())

            # 重命名列以匹配标准格式
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if 'date' in col_lower:
                    column_mapping[col] = 'Date'
                elif 'contest' in col_lower:
                    column_mapping[col] = 'Contest number'
                elif 'word' in col_lower:
                    column_mapping[col] = 'Word'
                elif 'reported' in col_lower:
                    column_mapping[col] = 'Number of reported results'
                elif 'hard mode' in col_lower:
                    column_mapping[col] = 'Number in hard mode'
                elif '1 try' in col_lower or '1 tries' in col_lower:
                    column_mapping[col] = '1 tries'
                elif '2 tries' in col_lower:
                    column_mapping[col] = '2 tries'
                elif '3 tries' in col_lower:
                    column_mapping[col] = '3 tries'
                elif '4 tries' in col_lower:
                    column_mapping[col] = '4 tries'
                elif '5 tries' in col_lower:
                    column_mapping[col] = '5 tries'
                elif '6 tries' in col_lower:
                    column_mapping[col] = '6 tries'
                elif '7 or more' in col_lower or 'x' in col_lower:
                    column_mapping[col] = '7 or more tries (X)'

            df = df.rename(columns=column_mapping)

            # 检查列名
            print("标准化后的列名:", df.columns.tolist())

        else:
            df = pd.read_csv(filepath, delimiter='\t')
            # 清理列名
            df.columns = [str(col).strip() for col in df.columns]

        print(f"数据形状: {df.shape}")

        # 查看数据前几行
        print("\n数据预览:")
        print(df.head())

        # 转换数值列
        numeric_columns = ['Number of reported results', 'Number in hard mode',
                           '1 tries', '2 tries', '3 tries', '4 tries',
                           '5 tries', '6 tries', '7 or more tries (X)']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 删除包含NaN的行
        df = df.dropna(subset=['Number of reported results'])

        # 计算平均猜测步数（加权平均）
        def calculate_avg_attempts(row):
            total = row['Number of reported results']
            if total == 0 or pd.isna(total):
                return 0

            attempts = 0
            weights = [1, 2, 3, 4, 5, 6, 7]  # 假设7次及以上计为7次

            for i, w in enumerate(weights, 1):
                if i <= 6:
                    col_name = f"{i} tries"
                else:
                    col_name = "7 or more tries (X)"

                if col_name in row:
                    value = row[col_name]
                    if pd.notna(value):
                        attempts += value * w
                    else:
                        attempts += 0

            return attempts / total

        df['avg_attempts'] = df.apply(calculate_avg_attempts, axis=1)

        # 计算成功率（6次及以内猜出）
        def calculate_success_rate(row):
            total = row['Number of reported results']
            if total == 0 or pd.isna(total):
                return 0

            success = 0
            for i in range(1, 7):
                col_name = f"{i} tries"
                if col_name in row:
                    value = row[col_name]
                    if pd.notna(value):
                        success += value
                    else:
                        success += 0

            return success / total

        df['success_rate'] = df.apply(calculate_success_rate, axis=1)

        # 计算困难模式比例
        if 'Number in hard mode' in df.columns and 'Number of reported results' in df.columns:
            df['hard_mode_ratio'] = df['Number in hard mode'] / df['Number of reported results']
        else:
            df['hard_mode_ratio'] = 0

        # 确保有Contest number列
        if 'Contest number' not in df.columns:
            df['Contest number'] = range(1, len(df) + 1)

        # 确保Date列是datetime格式
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # 如果没有Date列，创建一个从最早日期开始的时间序列
            start_date = pd.to_datetime('2022-12-31')
            df['Date'] = pd.date_range(start=start_date, periods=len(df), freq='D')

        print(f"\n处理后数据形状: {df.shape}")
        print(f"平均猜测步数范围: [{df['avg_attempts'].min():.2f}, {df['avg_attempts'].max():.2f}]")
        print(f"成功率范围: [{df['success_rate'].min():.2f}, {df['success_rate'].max():.2f}]")

        # 特征工程
        features, labels_attempts, labels_success = self._create_sequences(df)

        return features, labels_attempts, labels_success

    def _create_sequences(self, df):
        """创建时间序列数据"""
        features = []
        labels_attempts = []
        labels_success = []

        n_samples = len(df) - self.sequence_length
        if n_samples <= 0:
            raise ValueError(f"数据太少({len(df)}行)，无法创建长度为{self.sequence_length}的序列")

        print(f"创建时间序列，共有{n_samples}个样本")

        for i in range(n_samples):
            # 提取特征
            seq_features = []
            for j in range(self.sequence_length):
                idx = i + j
                feature_vector = []

                # 各种尝试次数的百分比
                for k in range(1, 7):
                    col_name = f"{k} tries"
                    if col_name in df.columns:
                        value = df.iloc[idx][col_name] / df.iloc[idx]['Number of reported results']
                        feature_vector.append(value)
                    else:
                        feature_vector.append(0)

                # 添加其他特征
                feature_vector.append(df.iloc[idx]['hard_mode_ratio'])
                feature_vector.append(df.iloc[idx]['Contest number'] / 1000)  # 归一化

                seq_features.append(feature_vector)

            features.append(seq_features)

            # 标签：下一轮的平均猜测步数和成功率
            next_idx = i + self.sequence_length
            labels_attempts.append(df.iloc[next_idx]['avg_attempts'])
            labels_success.append(1 if df.iloc[next_idx]['success_rate'] > 0.5 else 0)

        features = np.array(features)
        labels_attempts = np.array(labels_attempts)
        labels_success = np.array(labels_success)

        print(f"创建的特征形状: {features.shape}")
        print(f"标签数量: {len(labels_attempts)}")

        # 数据标准化
        original_shape = features.shape
        features_2d = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.fit_transform(features_2d)
        features = features_scaled.reshape(original_shape)

        return features, labels_attempts, labels_success

    def prepare_dataloaders(self, features, labels_attempts, labels_success, batch_size=32, test_size=0.2):
        """准备数据加载器"""
        # 不合并标签，保持分离状态
        X_train, X_test, y_train_attempts, y_test_attempts, y_train_success, y_test_success = train_test_split(
            features, labels_attempts, labels_success,
            test_size=test_size, random_state=42, shuffle=False
        )

        print(f"训练集大小: {len(X_train)}")
        print(f"测试集大小: {len(X_test)}")

        # 创建数据集
        train_dataset = WordleDataset(X_train, y_train_attempts, y_train_success, self.sequence_length)
        test_dataset = WordleDataset(X_test, y_test_attempts, y_test_success, self.sequence_length)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, y_test_attempts, y_test_success

    def simulate_data(self, n_days=365):
        """模拟生成示例数据"""
        print("正在生成模拟数据...")

        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

        data = {
            'Date': dates.strftime('%Y/%m/%d'),
            'Contest number': range(1, n_days + 1),
            'Word': ['word' + str(i) for i in range(n_days)],
            'Number of reported results': np.random.randint(15000, 25000, n_days),
            'Number in hard mode': np.random.randint(1000, 2500, n_days),
        }

        # 生成尝试次数的分布
        for i in range(1, 7):
            if i <= 3:
                data[f'{i} tries'] = np.random.randint(100, 500, n_days)
            elif i <= 5:
                data[f'{i} tries'] = np.random.randint(500, 2000, n_days)
            else:
                data[f'{i} tries'] = np.random.randint(2000, 5000, n_days)

        data['7 or more tries (X)'] = np.random.randint(100, 1000, n_days)

        df = pd.DataFrame(data)
        df.to_csv('wordle_data.csv', index=False, sep='\t')
        print(f"已生成模拟数据，共{n_days}天记录")

        return 'wordle_data.csv'