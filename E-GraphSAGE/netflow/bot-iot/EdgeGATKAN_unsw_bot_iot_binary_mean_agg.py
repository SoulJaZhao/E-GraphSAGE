import json
import os
import sys
import random
import socket
import struct
import warnings
from efficientKan import KANLinear

import category_encoders as ce
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import from_networkx
from dgl.nn import EdgeGATConv
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from torch.optim import Adam
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 定义保存 DGL 图的方法
def save_graph(graph, file_path):
    dgl.save_graphs(file_path, [graph])

# 定义加载 DGL 图的方法
def load_graph(file_path):
    return dgl.load_graphs(file_path)[0][0]

# 定义图文件路径
train_graph_file_path = 'binary_train_graph.dgl'
test_graph_file_path = 'binary_test_graph.dgl'
test_labels_file_path = 'binary_test_labels.npy'

# 定义分类报告文件路径
report_file_path = 'EdgeGATKAN_binary_classification_report.json'

best_model_file_path = 'EdgeGATKAN_binary_best_model.pth'

# 参数
epochs = 500

# 尝试加载训练图和测试图，如果文件不存在则创建图并保存
if os.path.exists(train_graph_file_path) and os.path.exists(test_graph_file_path):
    G = load_graph(train_graph_file_path)
    print("Train graph loaded from file.")
else:
    print("Train graph or test graph file not found. Creating new graph.")
    # 读取 CSV 文件到 DataFrame
    data = pd.read_csv('NF-BoT-IoT.csv')

    # 将 IPV4_SRC_ADDR 列中的每个 IP 地址替换为随机生成的 IP 地址
    # 这里生成的 IP 地址范围是从 172.16.0.1 到 172.31.0.1
    data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

    # 将 IPV4_SRC_ADDR 列中的每个值转换为字符串类型
    data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
    # 将 L4_SRC_PORT 列中的每个值转换为字符串类型
    data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
    # 将 IPV4_DST_ADDR 列中的每个值转换为字符串类型
    data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
    # 将 L4_DST_PORT 列中的每个值转换为字符串类型
    data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)

    # 将 IPV4_SRC_ADDR 和 L4_SRC_PORT 列的值连接起来，中间用冒号分隔
    data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
    # 将 IPV4_DST_ADDR 和 L4_DST_PORT 列的值连接起来，中间用冒号分隔
    data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']

    # 删除不再需要的 L4_SRC_PORT 和 L4_DST_PORT 列
    data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)

    # 删除不再需要的 Attack 列
    data.drop(columns=['Attack'], inplace=True)

    # 将 Label 列重命名为 label
    data.rename(columns={"Label": "label"}, inplace=True)

    # 将 label 列提取出来，保存到一个单独的变量中
    label = data.label

    # 从原始数据中删除 label 列
    data.drop(columns=['label'], inplace=True)

    # 创建 StandardScaler 对象，用于标准化数据
    scaler = StandardScaler()

    # 将 label 列重新加入到 data DataFrame 中，作为最后一列
    data = pd.concat([data, label], axis=1)

    # 将数据分为训练集和测试集，按 70% 和 30% 的比例分配，保证 stratify 参数确保按标签分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.3, random_state=2024, stratify=label)

    # 创建 TargetEncoder 对象，用于对分类特征进行目标编码
    encoder = ce.TargetEncoder(cols=['TCP_FLAGS', 'L7_PROTO', 'PROTOCOL'])

    # 用训练集的特征和标签拟合编码器
    encoder.fit(X_train, y_train)

    # 对训练集的特征进行编码转换
    X_train = encoder.transform(X_train)

    # 需要标准化的列，去除掉 label 列
    cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns)) - set(['label']))

    # 对需要标准化的列进行标准化
    X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])

    # 将标准化后的列组合成列表，添加为新的列 'h'
    X_train['h'] = X_train[cols_to_norm].values.tolist()

    # 从 pandas DataFrame 中创建一个无向多重图
    # 边的数据包含 'h' 和 'label' 列
    G = nx.from_pandas_edgelist(X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'], create_using=nx.MultiGraph())

    # 将无向图转换为有向图
    G = G.to_directed()

    # 将 NetworkX 图转换为 DGL 图，边的数据包含 'h' 和 'label' 属性
    G = from_networkx(G, edge_attrs=['h', 'label'])

    # 为每个节点的 'h' 属性赋值，初始值为全 1 的张量，维度与边的 'h' 属性相同
    G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

    # 为每条边添加 'train_mask' 属性，初始值为 True，表示这些边用于训练
    G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)

    # 保存图 G 到指定路径
    save_graph(G, train_graph_file_path)
    print("Train graph created and saved to file.")

# 定义计算准确度的函数
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

# 定义计算 F1 score 的函数
def compute_f1_score(pred, labels):
    pred_labels = pred.argmax(1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return f1_score(true_labels, pred_labels, average='weighted')

# 定义EdgeGAT模型
class EdgeGATModel(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = EdgeGATConv(
            in_feats=ndim_in,
            edge_feats=edim,
            out_feats=ndim_out,
            num_heads=8,
            activation=activation
        )

        self.pred = MLPPredictor(ndim_out, 2)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        h = h.mean(1)
        h = h.reshape(nfeats.shape[0], -1)
        return self.pred(g, h)

# 定义一个MLPPredictor类，继承自nn.Module
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        # 初始化MLPPredictor类
        # 定义线性层，输入维度为两倍的节点特征维度，输出维度为指定的类别数
        # self.W = nn.Linear(in_features * 2, out_classes)
        self.W = KANLinear(in_features * 2, out_classes)

    # 定义边应用函数，edges是DGL中的边数据
    def apply_edges(self, edges):
        # 获取源节点特征
        h_u = edges.src['h']
        # 获取目标节点特征
        h_v = edges.dst['h']
        # 将源节点和目标节点的特征连接起来，通过线性层转换
        score = self.W(th.cat([h_u, h_v], 1))
        # 返回包含预测得分的字典
        return {'score': score}

    # 定义前向传播函数
    def forward(self, graph, h):
        with graph.local_scope():
            # 使用local_scope保护当前图数据不被修改
            # 设置节点特征
            graph.ndata['h'] = h
            # 应用边上的计算函数，将预测得分存储在边数据中
            graph.apply_edges(self.apply_edges)
            # 返回边数据中的预测得分
            return graph.edata['score']

# 从图的边属性中提取标签并转换为 numpy 数组
edge_labels = G.edata['label'].cpu().numpy()

# 获取边标签的唯一值
unique_labels = np.unique(edge_labels)

# 计算每个类的权重，以处理类别不平衡问题
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=unique_labels,
                                                  y=edge_labels)

# 首先，根据是否有 CUDA 可用来设置设备
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 将 class_weights 转换为浮点张量，并根据设备选择移动到 GPU 或保持在 CPU 上
class_weights = th.FloatTensor(class_weights).to(device)

# 初始化损失函数，使用计算的类权重
criterion = nn.CrossEntropyLoss(weight=class_weights)

G = G.to(device)

# 获取节点特征和边特征
node_features = G.ndata['h']
edge_features = G.edata['h']

# 获取边标签和训练掩码
edge_label = G.edata['label']
train_mask = G.edata['train_mask']

# 将模型移动到设备上（GPU 或 CPU）
model = EdgeGATModel(G.ndata['h'].shape[1], 128, G.ndata['h'].shape[1], F.relu, 0.2).to(device)

# 将节点特征和边特征移动到设备上
node_features = node_features.to(device)
edge_features = edge_features.to(device)
edge_label = edge_label.to(device)
train_mask = train_mask.to(device)

# 定义优化器
opt = Adam(model.parameters())

# 变量用于保存最高的 F1 score
best_f1_score = 0.0

# 训练循环
for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
    # 前向传播，获取预测值
    pred = model(G, node_features, edge_features)

    # 计算损失，只考虑训练掩码内的边
    loss = criterion(pred[train_mask], edge_label[train_mask])

    # 清零梯度
    opt.zero_grad()

    # 反向传播，计算梯度
    loss.backward()

    # 更新模型参数
    opt.step()

    # 每 100 轮输出一次训练准确度和 F1 score
    if epoch % 100 == 0:
        accuracy = compute_accuracy(pred[train_mask], edge_label[train_mask])
        f1 = compute_f1_score(pred[train_mask], edge_label[train_mask])
        print(f'Epoch {epoch}: Training acc: {accuracy}, F1 score: {f1}')

    # 计算当前模型的 F1 score，如果高于最高的 F1 score，则保存模型和图
    current_f1_score = compute_f1_score(pred[train_mask], edge_label[train_mask])
    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        th.save(model, best_model_file_path)
        print(f'New best model and graph saved at epoch {epoch} with F1 score: {best_f1_score}')


if os.path.exists(test_graph_file_path):
    G_test = load_graph(test_graph_file_path)
    actual = np.load(test_labels_file_path)
    print("Test graph loaded from file.")
else:
    print("Test graph file not found. Creating new test graph.")
    # 对测试集进行目标编码转换
    X_test = encoder.transform(X_test)

    # 对需要标准化的列进行标准化
    X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

    # 将标准化后的列组合成列表，添加为新的列 'h'
    X_test['h'] = X_test[cols_to_norm].values.tolist()

    # 从 pandas DataFrame 中创建一个无向多重图
    # 边的数据包含 'h' 和 'label' 列
    G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h', 'label'],
                                     create_using=nx.MultiGraph())

    # 将无向图转换为有向图
    G_test = G_test.to_directed()

    # 将 NetworkX 图转换为 DGL 图，边的数据包含 'h' 和 'label' 属性
    G_test = from_networkx(G_test, edge_attrs=['h', 'label'])

    # 从 G_test 的边数据中取出 'label' 并删除
    actual = G_test.edata.pop('label')

    # 为 G_test 的每个节点设置 'feature' 属性，初始值为全 1 的张量，维度与训练图中的节点特征相同
    G_test.ndata['feature'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[2])

    # 保存测试图 G_test 到指定路径
    save_graph(G_test, test_graph_file_path)
    np.save(test_labels_file_path, actual)
    print("Test graph created and saved to file.")

# 将测试图移动到设备（GPU 或 CPU）
G_test = G_test.to(device)

import timeit

# 记录开始时间
start_time = timeit.default_timer()

# 获取测试图的节点特征和边特征
node_features_test = G_test.ndata['feature']
edge_features_test = G_test.edata['h']

# 进行前向传播，获取测试预测
best_model = th.load(best_model_file_path)
best_model = best_model.to(device)
best_model.eval()
test_pred = best_model(G_test, node_features_test, edge_features_test).to(device)

# 计算并打印前向传播所花费的时间
elapsed = timeit.default_timer() - start_time
print(str(elapsed) + ' seconds')

# 获取预测标签
test_pred = test_pred.argmax(1)

# 将预测结果从 GPU 移动到 CPU，并转换为 numpy 数组
test_pred = test_pred.cpu().detach().numpy()

# 将实际标签和预测标签转换为 "Normal" 或 "Attack"
actual = ["Normal" if i == 0 else "Attack" for i in actual]
test_pred = ["Normal" if i == 0 else "Attack" for i in test_pred]

# 打印详细的分类报告
report = classification_report(actual, test_pred, target_names=["Normal", "Attack"], output_dict=True)
# 保存分类报告为JSON文件
with open(report_file_path, 'w') as jsonfile:
    json.dump(report, jsonfile, indent=4)

print(classification_report(actual, test_pred, target_names=["Normal", "Attack"]))

# 定义绘制混淆矩阵的函数
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}\n'
               f'Precision={report["weighted avg"]["precision"]:0.4f}; Recall={report["weighted avg"]["recall"]:0.4f}; F1-Score={report["weighted avg"]["f1-score"]:0.4f}')
    plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(actual, test_pred)
plot_confusion_matrix(cm=cm,
                      normalize=False,
                      target_names=np.unique(actual),
                      title="Confusion Matrix")