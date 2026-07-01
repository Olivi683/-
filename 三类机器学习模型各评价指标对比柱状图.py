import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# ========== 1. 自动加载同目录下的中文字体文件 ==========
def load_font():
    for file in os.listdir('.'):
        if file.endswith(('.otf', '.ttf')):
            font_path = os.path.join('.', file)
            print(f"✅ 找到字体文件: {font_path}，将使用此字体")
            return FontProperties(fname=font_path, size=12)
    # 如果没找到，尝试系统字体（可能乱码，但至少不报错）
    try:
        fp = FontProperties(family='SimHei', size=12)
        print("⚠️ 未找到字体文件，尝试使用系统黑体，可能仍乱码")
        return fp
    except:
        raise RuntimeError("❌ 未找到任何中文字体文件！请将 .otf 或 .ttf 字体文件放在脚本同目录。")

font_prop = load_font()

# ========== 2. 数据 ==========
models = ['随机森林', 'SVM', '逻辑回归']
metrics = ['准确率', '精确率', '召回率', 'F1分数']
data = {
    '准确率':  [0.928, 0.916, 0.952],
    '精确率': [0.879, 0.958, 0.895],
    '召回率': [0.920, 0.793, 0.977],
    'F1分数': [0.899, 0.868, 0.934]
}
df = pd.DataFrame(data, index=models)

# ========== 3. 绘图 ==========
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(metrics))
width = 0.25
colors = ['#FFB7C5', '#A8E6CF', '#D4A5FF']  # 马卡龙色

for i, model in enumerate(models):
    vals = df.loc[model].values
    bars = ax.bar(x + i*width, vals, width, label=model, color=colors[i], edgecolor='black', linewidth=0.8)
    if model == 'SVM':
        bars[2].set_color('#E74C3C')   # SVM召回率标红
        bars[2].set_edgecolor('darkred')

ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontproperties=font_prop, fontsize=12)
ax.set_ylabel('性能得分', fontproperties=font_prop, fontsize=12)
ax.set_ylim(0.70, 1.01)
legend = ax.legend(loc='lower right', prop=font_prop, fontsize=11)
# ⭐ 标题已修改，去掉了“（马卡龙配色）”
ax.set_title('各模型性能指标对比', fontproperties=font_prop, fontweight='bold', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figure1_performance_bars.png', dpi=300)
plt.show()
