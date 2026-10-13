from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

folder = Path(__file__).resolve().parent
results_dir = folder / 'results'
figures_dir = folder / 'figures'
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# ---------- 1. 数据读取、探索与划分 ----------
train_df = pd.read_csv(folder / 'train_data.csv')
if train_df[['text', 'target']].isna().any().any():
    raise ValueError('训练数据存在缺失值，请先检查数据')

print('训练数据形状：', train_df.shape)
print('各类别样本数量：\n', train_df['target'].value_counts().sort_index())
print('文本长度统计：\n', train_df['text'].astype(str).str.len().describe())

X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'].astype(str).tolist(),
    train_df['target'].to_numpy(),
    test_size=0.2, random_state=42, stratify=train_df['target']
)
print('训练集：', len(X_train), '验证集：', len(X_val))

# ---------- 2. TF-IDF 特征提取 ----------
# 只用训练集学习词表，验证集只做 transform，避免数据泄露
vectorizer = TfidfVectorizer(max_features=5000, lowercase=True,
                             stop_words=None, ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
print('TF-IDF矩阵：', X_train_tfidf.shape, X_val_tfidf.shape)

# 两个小函数只负责保存表格和绘图，模型训练仍使用普通for循环

def save_results(filename, rows):
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / filename, index=False)
    print('\n' + filename + '\n' + df.to_string(index=False))
    return df


def draw_curve(df, x_name, filename, log_scale=False, show_train=False):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_name], df['Macro-F1'], 'o-', label='Validation Macro-F1')
    if show_train:
        plt.plot(df[x_name], df['Train Macro-F1'], 's-', label='Train Macro-F1')
    else:
        plt.plot(df[x_name], df['Accuracy'], 's-', label='Validation Accuracy')
    if log_scale:
        plt.xscale('log')
    plt.xlabel(x_name)
    plt.ylabel('Score')
    plt.title(filename.replace('_', ' ').replace('.png', ''))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=250)
    plt.close()


# ---------- 3. MLP默认实验配置 ----------
def create_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(100,), activation='relu', solver='adam',
        alpha=0.0001, learning_rate_init=0.001, max_iter=100,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42
    )


# ---------- 4. 基线与参数实验 ----------
quick = '--quick' in sys.argv

if not quick:
    baseline_models = [
        ('MultinomialNB', MultinomialNB(alpha=1.0)),
        ('LogisticRegression', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ('SVC_linear', SVC(kernel='linear', C=1.0, random_state=42)),
        ('MLP', create_mlp())
    ]
    baseline_rows = []
    for name, model in baseline_models:
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        baseline_rows.append({
            'Model': name, 'Accuracy': accuracy_score(y_val, pred),
            'Macro-F1': f1_score(y_val, pred, average='macro'),
            'Train Time (s)': seconds
        })
    save_results('baseline_results.csv', baseline_rows)

    # 4.1 朴素贝叶斯 alpha
    nb_rows = []
    for alpha in [0.01, 0.1, 0.5, 1, 2, 5, 10]:
        model = MultinomialNB(alpha=alpha)
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        nb_rows.append({'alpha': alpha, 'Accuracy': accuracy_score(y_val, pred),
                        'Macro-F1': f1_score(y_val, pred, average='macro'),
                        'Train Time (s)': seconds})
    df = save_results('nb_tuning_results.csv', nb_rows)
    draw_curve(df, 'alpha', 'nb_tuning_results.png', log_scale=True)

    # 4.2 逻辑回归 C
    lr_rows = []
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=2000, random_state=42)
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        lr_rows.append({'C': C, 'Accuracy': accuracy_score(y_val, pred),
                        'Macro-F1': f1_score(y_val, pred, average='macro'),
                        'Train Time (s)': seconds})
    df = save_results('lr_tuning_results.csv', lr_rows)
    draw_curve(df, 'C', 'lr_tuning_results.png', log_scale=True)

    # 4.3 线性SVM C。注意：LinearSVC 与上面的 SVC(kernel='linear') 实现不同
    svm_rows = []
    for C in [0.01, 0.1, 1, 10, 100]:
        model = LinearSVC(C=C, max_iter=5000, random_state=42)
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        svm_rows.append({'C': C, 'Accuracy': accuracy_score(y_val, pred),
                         'Macro-F1': f1_score(y_val, pred, average='macro'),
                         'Train Time (s)': seconds})
    df = save_results('svm_tuning_results.csv', svm_rows)
    draw_curve(df, 'C', 'svm_tuning_results.png', log_scale=True)

    # 4.4 MLP 隐藏层结构：只改变 hidden_layer_sizes
    structure_rows = []
    for structure in [(50,), (100,), (200,), (100, 50)]:
        model = MLPClassifier(
            hidden_layer_sizes=structure, activation='relu', solver='adam',
            alpha=0.0001, learning_rate_init=0.001, max_iter=100,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42)
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        train_f1 = f1_score(y_train, model.predict(X_train_tfidf), average='macro')
        val_f1 = f1_score(y_val, pred, average='macro')
        structure_rows.append({'Structure': str(structure), 'Accuracy': accuracy_score(y_val, pred),
                               'Macro-F1': val_f1, 'Train Macro-F1': train_f1,
                               'F1 Gap': train_f1 - val_f1, 'Iterations': model.n_iter_,
                               'Train Time (s)': seconds})
    df = save_results('mlp_structure_tuning.csv', structure_rows)
    draw_curve(df, 'Structure', 'mlp_structure_tuning.png', show_train=True)

    # 4.5 MLP 正则化 alpha：只改变 alpha
    alpha_rows = []
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        model = MLPClassifier(
            hidden_layer_sizes=(100,), activation='relu', solver='adam',
            alpha=alpha, learning_rate_init=0.001, max_iter=100,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42)
        start = time.perf_counter()
        model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        train_f1 = f1_score(y_train, model.predict(X_train_tfidf), average='macro')
        val_f1 = f1_score(y_val, pred, average='macro')
        alpha_rows.append({'Alpha': alpha, 'Accuracy': accuracy_score(y_val, pred),
                           'Macro-F1': val_f1, 'Train Macro-F1': train_f1,
                           'F1 Gap': train_f1 - val_f1, 'Iterations': model.n_iter_,
                           'Train Time (s)': seconds})
    df = save_results('mlp_alpha_tuning.csv', alpha_rows)
    draw_curve(df, 'Alpha', 'mlp_alpha_tuning.png', log_scale=True, show_train=True)

    # 4.6 MLP训练轮数：关闭早停，以便比较10、20、40、80轮
    iteration_rows = []
    for max_iter in [10, 20, 40, 80]:
        model = MLPClassifier(
            hidden_layer_sizes=(100,), activation='relu', solver='adam',
            alpha=0.0001, learning_rate_init=0.001, max_iter=max_iter,
            early_stopping=False, tol=0.0,
            n_iter_no_change=max_iter + 1, random_state=42)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model.fit(X_train_tfidf, y_train)
        seconds = time.perf_counter() - start
        pred = model.predict(X_val_tfidf)
        train_f1 = f1_score(y_train, model.predict(X_train_tfidf), average='macro')
        val_f1 = f1_score(y_val, pred, average='macro')
        iteration_rows.append({'Max Iter': max_iter, 'Actual Iter': model.n_iter_,
                               'Accuracy': accuracy_score(y_val, pred), 'Macro-F1': val_f1,
                               'Train Macro-F1': train_f1, 'F1 Gap': train_f1 - val_f1,
                               'Train Time (s)': seconds})
    df = save_results('mlp_iteration_tuning.csv', iteration_rows)
    draw_curve(df, 'Max Iter', 'mlp_iteration_tuning.png', show_train=True)

# ---------- 5. 各模型最佳已测配置比较 ----------
best_models = [
    ('Naive Bayes', MultinomialNB(alpha=0.1)),
    ('Logistic Regression', LogisticRegression(C=100, max_iter=2000, random_state=42)),
    ('LinearSVC', LinearSVC(C=1, max_iter=5000, random_state=42)),
    ('MLP', create_mlp())
]
comparison_rows = []
for name, model in best_models:
    print('\n正在训练：', name)
    start = time.perf_counter()
    model.fit(X_train_tfidf, y_train)
    seconds = time.perf_counter() - start
    pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average='macro')
    comparison_rows.append({'Model': name, 'Accuracy': acc,
                            'Macro-F1': f1, 'Train Time (s)': seconds})
    print('Accuracy:', round(acc, 6), 'Macro-F1:', round(f1, 6))

    if name == 'MLP':
        print('\nMLP分类报告：\n', classification_report(y_val, pred))
        # ---------- 6. MLP混淆矩阵与误分类样本 ----------
        labels = np.unique(y_val)
        cm = confusion_matrix(y_val, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title('MLP Confusion Matrix')
        fig.tight_layout()
        fig.savefig(figures_dir / 'mlp_confusion_matrix.png', dpi=250)
        plt.close(fig)

        confusion_rows = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    confusion_rows.append({'True': labels[i], 'Predicted': labels[j],
                                           'Count': int(cm[i, j])})
        confusion_rows.sort(key=lambda row: row['Count'], reverse=True)
        save_results('top_confusions.csv', confusion_rows)

        wrong_indices = np.where(y_val != pred)[0]
        wrong_rows = []
        for i in wrong_indices[:20]:
            wrong_rows.append({'True': y_val[i], 'Predicted': pred[i],
                               'Text excerpt': X_val[i][:500]})
        save_results('misclassified_samples.csv', wrong_rows)
        print('MLP误分类：', len(wrong_indices), '/', len(y_val))

comparison_df = save_results('best_model_comparison.csv', comparison_rows)
plt.figure(figsize=(8, 5))
plt.bar(comparison_df['Model'], comparison_df['Macro-F1'])
plt.ylabel('Validation Macro-F1')
plt.ylim(0.80, 0.95)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(figures_dir / 'best_model_comparison.png', dpi=250)
plt.close()

print('\n实验完成！CSV在 results/，图片在 figures/。')
