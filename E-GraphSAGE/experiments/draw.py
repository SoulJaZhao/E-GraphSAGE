import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 模型名称和对应的平均指标
data = {
    'Model': ['GAT', 'E-GraphSAGE', 'Anormal-E', 'SCENE', 'SKGFusionKAN'],
    'F1 Score': [73.38, 83.05, 81.03, 74.88, 85.15],
    'Precision': [74.25, 87.22, 81.77, 84.60, 88.23],
    'Recall': [73.72, 81.48, 81.48, 74.81, 83.89]
}

df = pd.DataFrame(data)

# 设置字体为 Times New Roman
plt.rc('font', family='Times New Roman')

# 适合SCI顶会论文的配色
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC']

def plot_metric(metric, title):
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x='Model', y=metric, data=df, palette=colors, width=0.6)

    # 在每个柱子上显示数值
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=28)

    # 绘制折线
    plt.plot(df['Model'], df[metric], marker='o', color='red', linewidth=2, label='Trend')

    # 移除上边和右边的边框
    # ax = plt.gca()  # 获取当前的坐标轴
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # 设置Y轴的范围为0到100
    plt.ylim(0, 100)

    plt.title('', fontsize=30)
    plt.xlabel('', fontsize=30)
    plt.ylabel(title, fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # 添加图例到底部
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=df['Model'][i]) for i in range(len(df))]
    # plt.legend(handles=handles, fontsize=15, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.savefig(f'average_{metric.lower()}_comparison_with_trend.png', dpi=400)
    plt.close()

# 绘制三种指标的图表
plot_metric('F1 Score', 'Average F1 Score (%)')
plot_metric('Precision', 'Average Precision (%)')
plot_metric('Recall', 'Average Recall (%)')
