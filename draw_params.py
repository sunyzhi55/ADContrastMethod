import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Resnet:flops: 70.924G, params: 66.952M
ViT:flops: 5.775G, params: 20.853M
EfficientNet:flops: 75.563M, params: 1.638M
Poolformer:flops: 9.549G, params: 31.210M
nnMamba:flops: 48.626G, params: 26.042M
Diamond:flops: 97.638G, params: 23.504M
VAPL:flops: 40.350G, params: 63.024M
IMF:flops: 70.925G, params: 67.843M
HFBSurv:flops: 141.849G, params: 34.123M
ITCFN:flops: 71.098G, params: 71.305M
HyperFusionNet:flops: 47.750G, params: 15.402M
MDL:flops: 9.353G, params: 2.827M
MDL_two_inputs:flops: 19.243G, params: 10.707M
RLAD:flops: 260.882G, params: 30.624M
AweSomeNet:flops: 10.517G, params: 17.405M
"""
# 数据准备
data = {
    "Model": ["Resnet", "ViT", "EfficientNet", "Poolformer", "nnMamba", "VAPL",
              "IMF", "HFBSurv", "ITCFN", "HyperFusionNet", "MDL",  "AweSomeNet", "Diamond"],
    "FLOPs (G)": [70.924, 5.775, 0.0756, 9.549, 48.626, 40.350,
                  70.925, 141.849, 71.098, 47.750, 19.243, 10.517, 97.638],
    "Params (M)": [66.952, 20.853, 1.638, 31.210, 26.042, 63.024,
                   67.843, 34.123, 71.305, 15.402, 10.707, 17.405, 23.504]
}
df = pd.DataFrame(data)

# 绘制气泡图
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=df["FLOPs (G)"],
    y=df["Params (M)"],
    s=df["Params (M)"] * 2,
    c=np.log10(df["FLOPs (G)"]),
    cmap="viridis",
    alpha=0.7
)

# 添加标签和装饰
for i, row in df.iterrows():
    plt.text(row["FLOPs (G)"], row["Params (M)"], row["Model"],
             fontsize=8, ha='center', va='bottom')
plt.colorbar(scatter, label="log10(FLOPs) (G)")
plt.xlabel("FLOPs (Giga)")
plt.ylabel("Parameters (Millions)")
plt.title("Model Complexity: FLOPs vs. Parameters")
plt.grid(linestyle='--')

# 保存图片（调整dpi提高清晰度）
plt.savefig("model_comparison_bubble.png", dpi=500, bbox_inches='tight')  # PNG格式
plt.savefig("model_comparison_bubble.pdf", bbox_inches='tight')          # PDF格式（矢量图）
plt.savefig("model_comparison_bubble.svg", bbox_inches='tight')          # SVG格式（矢量图）

print("图片已保存为: model_comparison_bubble.png/pdf/svg")

import plotly.express as px

# fig = px.scatter(
#     df, x="FLOPs (G)", y="Params (M)",
#     size="Params (M)", color="FLOPs (G)",
#     hover_name="Model", log_x=True,
#     title="Interactive Model Comparison"
# )
fig = px.scatter(
    df,
    x="FLOPs (G)",
    y="Params (M)",
    size="Params (M)",
    color="FLOPs (G)",
    hover_name="Model",
    text="Model",  # 关键修改：直接使用Model列作为标签
    log_x=True,
    title="Interactive Model Comparison (Circle Size = Params, Color = FLOPs)"
)
# 调整标签样式（避免重叠）
fig.update_traces(
    textposition='top center',  # 标签位置：top, bottom, left, right等
    textfont_size=10,          # 字体大小
    marker=dict(opacity=0.7)   # 圆点透明度
)
# 保存为HTML（可交互）
fig.write_html("interactive_model_comparison.html")

# 保存为静态图片（需安装kaleido）
fig.write_image("interactive_model_comparison.png", engine="kaleido", scale=2)


plt.figure(figsize=(10, 10))
for i, row in df.iterrows():
    plt.scatter([i], [i],
                s=row["Params (M)"] * 5,
                c=np.log10(row["FLOPs (G)"]),
                cmap="plasma",
                alpha=0.6)
    plt.text(i, i, f"{row['Model']}\nParams: {row['Params (M)']}M\nFLOPs: {row['FLOPs (G)']}G",
             ha='center', va='center', fontsize=8)

plt.colorbar(label="log10(FLOPs) (G)")
plt.axis('off')
plt.title("Model Complexity (Size=Params, Color=FLOPs)")

plt.savefig("model_comparison_circle.jpg", dpi=300, transparent=True)  # 透明背景
