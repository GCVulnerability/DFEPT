import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ["sum", "max", "mean", "uni"]
ggnn_acc = [0.6274, 0.6391, 0.6369, 0.6318]  # Devign 数据集的准确率 (GGNN)
gcn_acc = [0.6402, 0.638, 0.6354, 0.6497]    # Devign 数据集的准确率 (GCN)
ggnn_acc_reveal = [0.9099, 0.9116, 0.9116, 0.9085]  # Reveal 数据集的准确率 (GGNN)
gcn_acc_reveal = [0.9195, 0.9103, 0.9103, 0.9182]   # Reveal 数据集的准确率 (GCN)
x = np.arange(len(models))  # 横坐标位置
bar_width = 0.35  # 柱状图的宽度

plt.rcParams.update({'font.size': 32})

# 创建一个 30x12 英寸的画布
plt.figure(figsize=(22, 12))

# Devign 数据集的柱状图
plt.subplot(1, 2, 1)  # 1行2列，当前是第1个图
plt.bar(x - bar_width/2, ggnn_acc, bar_width, label='GGNN', color='#B5A1E3', edgecolor='black')
plt.bar(x + bar_width/2, gcn_acc, bar_width, label='GCN', color='#F0C2A2', edgecolor='black')
plt.xlabel('Pooling Method', fontsize=32)
plt.ylabel('Accuracy', fontsize=32)
plt.title('Devign', fontsize=40)
plt.xticks(x, models)
plt.ylim(0.62, 0.66)
plt.legend(prop={'size': 30})

# Reveal 数据集的柱状图
plt.subplot(1, 2, 2)  # 1行2列，当前是第2个图
plt.bar(x - bar_width/2, ggnn_acc_reveal, bar_width, label='GGNN', color='#B5A1E3', edgecolor='black')
plt.bar(x + bar_width/2, gcn_acc_reveal, bar_width, label='GCN', color='#F0C2A2', edgecolor='black')
plt.xlabel('Pooling Method', fontsize=32)
plt.title('Reveal', fontsize=40)
plt.xticks(x, models)
plt.ylim(0.90, 0.93)
plt.legend(prop={'size': 30})

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
# 显示整个画布
plt.savefig(f"Fig3.pdf", dpi=200, bbox_inches='tight')
plt.show()
