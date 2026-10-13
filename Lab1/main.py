import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# 获取当前代码所在的文件夹
BASE_DIR = Path(__file__).resolve().parent


def load_data():
    # 读取训练集和无标签测试集
    train_data = pd.read_csv(BASE_DIR / 'train_data.csv')
    test_data = pd.read_csv(BASE_DIR / 'test_data_unlabeled.csv')

    return train_data, test_data


def main():
    # 1. 读取数据
    train_data, test_data = load_data()

    print("训练集样本数：", len(train_data))
    print("测试集样本数：", len(test_data))

    # 2. TF-IDF特征提取
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])

    # 3. 创建MLP模型
    # 根据验证集实验，选择隐藏层为100个神经元的MLP
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )

    # 4. 使用全部有标签数据训练模型
    print("正在训练MLP模型...")
    model.fit(X_train, train_data['target'])

    # 5. 预测测试集
    predictions = model.predict(X_test)

    # 6. 保存预测结果
    output_path = BASE_DIR / 'prediction.csv'
    pd.DataFrame(predictions).to_csv(
        output_path,
        index=False,
        header=False
    )

    print("实际训练轮数：", model.n_iter_)
    print("预测样本数：", len(predictions))
    print("预测结果已保存：", output_path)


if __name__ == '__main__':
    main()
