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
    try:
        fp = FontProperties(family='SimHei', size=12)
        print("⚠️ 未找到字体文件，尝试使用系统黑体，可能仍乱码")
        return fp
    except:
        raise RuntimeError("❌ 未找到任何中文字体文件！请将 .otf 或 .ttf 字体文件放在脚本同目录。")

font_prop = load_font()

# ========== 2. 数据 ==========
# ⚠️ 这里用的是假设分布，请替换为您的真实数据！
# 三个风险等级的比例（加起来必须等于100）
risk_labels = ['低风险\n(<0.1)', '中风险\n(0.1~0.8)', '高风险\n(≥0.8)']
risk_sizes = [10, 80, 10]  # 分别是低、中、高风险的占比（%）
risk_colors = ['#B5EAD7', '#FFDAC1', '#FF9AA2']  # 马卡龙绿、杏、粉（红绿灯风格）

print(f"当前使用的风险分布：低风险 {risk_sizes[0]}%，中风险 {risk_sizes[1]}%，高风险 {risk_sizes[2]}%")
print("⚠️ 请替换为您的真实数据！")

# ========== 3. 绘制环形图 ==========
fig, ax = plt.subplots(figsize=(8, 7))

# 绘制圆环图（宽高比1:1保证正圆）
wedges, texts, autotexts = ax.pie(
    risk_sizes, 
    labels=risk_labels, 
    colors=risk_colors,
    autopct='%1.0f%%',           # 显示百分比
    startangle=90,               # 从12点钟方向开始
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),  # width控制环的粗细
    textprops={'fontsize': 13}
)

# 让百分比文字使用中文字体
for autotext in autotexts:
    autotext.set_fontproperties(font_prop)
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

# 让标签文字使用中文字体
for text in texts:
    text.set_fontproperties(font_prop)
    text.set_fontsize(13)

# ========== 4. 标题和注释 ==========
ax.set_title('风险分层分布', fontproperties=font_prop, fontweight='bold', fontsize=16)

# 底部添加策略说明（金字塔预警逻辑）
ax.text(
    0.5, -0.15, 
    '策略：中风险由AI自动处理，高风险需要人工介入', 
    transform=ax.transAxes, 
    ha='center', 
    fontproperties=font_prop, 
    fontsize=11, 
    color='gray', 
    style='italic'
)

plt.tight_layout()
plt.savefig('figure5_risk_pie.png', dpi=300)
plt.show()
