import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import os

# ========== 1. 自动加载同目录下的中文字体文件 ==========
def load_font():
    for file in os.listdir('.'):
        if file.endswith(('.otf', '.ttf')):
            font_path = os.path.join('.', file)
            print(f"✅ 找到字体文件: {font_path}，将使用此字体")
            return FontProperties(fname=font_path, size=11)
    try:
        fp = FontProperties(family='SimHei', size=11)
        print("⚠️ 未找到字体文件，尝试使用系统黑体，可能仍乱码")
        return fp
    except:
        raise RuntimeError("❌ 未找到任何中文字体文件！请将 .otf 或 .ttf 字体文件放在脚本同目录。")

font_prop = load_font()

# ========== 2. 数据（完全来自您的表格，真实数据） ==========
# 特征名称（翻译为中文，更直观）
features = [
    '抑郁评分', 
    '焦虑评分', 
    '焦虑×生理指标', 
    '社交×抑郁', 
    '焦虑/抑郁比值',
    '压力评分', 
    '压力-抑郁差距', 
    '社交孤立风险', 
    '学业×心理交互', 
    '社交媒体使用时长'
]
# 对应的重要性分数
importances = [0.2393, 0.1335, 0.1034, 0.0764, 0.0547, 0.0517, 0.0452, 0.0396, 0.0279, 0.0164]

# ========== 3. 绘图 ==========
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))

# 颜色映射：前2个深红色，第3-4个橙色（马卡龙杏色），其余灰色
colors = []
for i in range(len(features)):
    if i < 2:
        colors.append('#C0392B')          # 深红色（抑郁+焦虑）
    elif i < 4:
        colors.append('#FFDAC1')          # 马卡龙杏色（交互项）
    else:
        colors.append('#95A5A6')          # 灰色（其余）

# 绘制水平条形图
bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', linewidth=0.8, height=0.6)

# 设置Y轴标签（特征名称）
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontproperties=font_prop, fontsize=11)
ax.invert_yaxis()  # 重要：让第一条在顶部

# X轴和标题
ax.set_xlabel('重要性得分', fontproperties=font_prop, fontsize=12)
ax.set_title('特征重要性排名（前10）', fontproperties=font_prop, fontweight='bold', fontsize=14)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# ========== 4. 在右侧标注累计贡献百分比 ==========
cumsum = np.cumsum(importances)
for i, (bar, cum) in enumerate(zip(bars, cumsum)):
    ax.text(
        bar.get_width() + 0.005,          # X位置：条形右侧稍远一点
        bar.get_y() + bar.get_height()/2, # Y位置：条形中间
        f'{cum:.1%}',                     # 显示累计百分比（如 23.9%）
        va='center', 
        fontproperties=font_prop, 
        fontsize=9, 
        fontweight='bold',
        color='black'
    )

# ========== 5. 添加图例 ==========
legend_elements = [
    Patch(facecolor='#C0392B', label='前两名（抑郁+焦虑）'),
    Patch(facecolor='#FFDAC1', label='交互项（橙色）'),
    Patch(facecolor='#95A5A6', label='其余特征（灰色）')
]
ax.legend(handles=legend_elements, prop=font_prop, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_feature_importance.png', dpi=300)
plt.show()
