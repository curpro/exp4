import numpy as np
import matplotlib.pyplot as plt
import os

# 确保路径正确
file_path = './npz_pos/training_data_part0.npz'

if not os.path.exists(file_path):
    print(f"错误：找不到文件 {file_path}")
else:
    data = np.load(file_path)
    if 'Y' not in data:
        print("错误：数据文件中没有找到 'Y' 数组")
    else:
        # 原始数据 Shape: (N, 6) -> [P11, P12, P21, P22, P31, P32]
        Y_raw = data['Y']
        N = Y_raw.shape[0]
        print(f"成功加载数据，样本数: {N}")

        # --- 核心改进：计算缺失的第三列概率 (P_i3) ---
        # 概率守恒：Row Sum = 1
        # P13 = 1 - (P11 + P12)
        P13 = 1.0 - (Y_raw[:, 0] + Y_raw[:, 1])
        # P23 = 1 - (P21 + P22)
        P23 = 1.0 - (Y_raw[:, 2] + Y_raw[:, 3])
        # P33 = 1 - (P31 + P32)  <-- 战斗机持续转弯的关键
        P33 = 1.0 - (Y_raw[:, 4] + Y_raw[:, 5])

        # 为了防止浮点数误差导致微小的负数，截断到 [0, 1]
        P13 = np.clip(P13, 0, 1)
        P23 = np.clip(P23, 0, 1)
        P33 = np.clip(P33, 0, 1)

        # 重新组合成完整的 (N, 9) 数据，方便按 3x3 矩阵绘图
        # 顺序: Row1(CV), Row2(CA), Row3(CT)
        Y_full = np.column_stack((
            Y_raw[:, 0], Y_raw[:, 1], P13,  # From CV: To CV, CA, CT
            Y_raw[:, 2], Y_raw[:, 3], P23,  # From CA: To CV, CA, CT
            Y_raw[:, 4], Y_raw[:, 5], P33  # From CT: To CV, CA, CT
        ))

        # --- 绘图设置 ---

        # 标签矩阵
        labels = [
            r'$P_{11}$ (CV$\to$CV)', r'$P_{12}$ (CV$\to$CA)', r'$P_{13}$ (CV$\to$CT)',
            r'$P_{21}$ (CA$\to$CV)', r'$P_{22}$ (CA$\to$CA)', r'$P_{23}$ (CA$\to$CT)',
            r'$P_{31}$ (CT$\to$CV)', r'$P_{32}$ (CT$\to$CA)', r'$P_{33}$ (CT$\to$CT)'
        ]

        # 颜色定义：Row 1 (Blue), Row 2 (Orange), Row 3 (Green)
        # 代表 "From State" 的颜色
        colors = ['blue'] * 3 + ['orange'] * 3 + ['green'] * 3

        # 创建 3行 x 3列 的子图 (标准的 TPM 矩阵结构)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        for i in range(9):
            ax = axes[i]
            data_col = Y_full[:, i]

            # 绘制直方图
            n, bins, patches = ax.hist(data_col, bins=50, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)

            # 样式调整
            ax.set_title(labels[i], fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, linestyle='--', alpha=0.3)

            # 只有最左侧显示 Y 轴标签，最底部显示 X 轴标签，保持整洁
            if i % 3 == 0:
                ax.set_ylabel("Count", fontsize=12)
            if i >= 6:
                ax.set_xlabel("Probability", fontsize=12)

            # 计算统计量
            mean_val = np.mean(data_col)
            std_val = np.std(data_col)

            # 绘制均值线
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')

            # 在右上角显示更详细的统计信息
            stats_text = f"Mean: {mean_val:.3f}\nStd:  {std_val:.3f}"
            ax.text(0.95, 0.9, stats_text, transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 如果是对角线元素 (P11, P22, P33)，加粗边框提示重点
            if i in [0, 4, 8]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)

        plt.suptitle(
            f"Full TPM Distribution for Fighter Jet Tracking (Samples: {N})\n(Highlight: Diagonal = Model Retention)",
            fontsize=18, y=0.98)
        plt.tight_layout()
        plt.show()