# mnist_tq.py 高层流程 (High-Level Flow)

该文件是基于量子神经网络 (QNet) 对 MNIST 数据集进行概念解释 (Concept Attribution) 的实验脚本。其核心逻辑围绕 **CAR (Concept Activation Regions)** 和 **CAV (Concept Activation Vectors)** 展开。

## 整体流程概述

### 1. 初始化与配置
*   **参数设置**：通过 `Arguments` 类初始化任务配置（如 `MNIST_10`）。
*   **量子电路设计**：设计量子电路的架构，包括单比特门 (`single`) 和受控门 (`enta`) 的布局。
*   **概念定义**：定义了四个用于解释的概念及其对应的 MNIST 数字类别：
    *   `Loop` (环): [0, 2, 6, 8, 9]
    *   `Vertical Line` (垂直线): [1, 4, 7]
    *   `Horizontal Line` (水平线): [4, 5, 7]
    *   `Curvature` (曲率): [0, 2, 3, 5, 6, 8, 9]

### 2. 模型准备 (`train_mnist_model`)
*   构建 `QNet` 模型（基于 `torchquantum`）。
*   训练或从本地加载预训练的 `vqc_model.pt` 模型权重。

### 3. 概念准确度实验 (`concept_accuracy`)
*   **数据生成**：针对每个定义的概念生成平衡的正负样本数据集。
*   **特征提取**：利用 `hooks` 机制提取模型中间层的表示（Representations）。
*   **分类器训练**：在提取的特征空间上训练 `CAR` 和 `CAV` 分类器。
*   **评估**：计算并保存分类器在训练集和测试集上的准确率，以验证潜在空间是否编码了这些人类可理解的概念。

### 4. 统计显著性测试 (`statistical_significance`)
*   对概念分类器的性能进行统计验证，确保提取的概念具有代表性而非随机巧合。

### 5. 全局解释生成 (`global_explanations`)
*   利用训练好的 `CAR` 分类器，分析模型在全局范围内对各类别预测时所依赖的主要概念。

### 6. 特征重要性分析 (`feature_importance`)
*   **CAR vs Vanilla**：对比基于 CAR 的概念重要性归因与传统的特征重要性方法（如 Vanilla 梯度）。
*   **可视化**：生成归因相关性图表。

### 7. 核函数敏感性分析 (`kernel_sensitivity`)
*   研究 CAR 方法在不同核函数（如 Matern 核）下的表现和稳定性。

### 8. 概念集规模影响分析 (`concept_size_impact`)
*   评估训练概念分类器时使用的样本数量（Concept Set Size）对最终解释结果的影响。

### 9. 概念间关系分析 (`tcar_inter_concept`)
*   利用 TCAR 指标分析不同概念之间的相互作用或重叠程度。

### 10. 对抗鲁棒性测试 (`adversarial_robustness`)
*   **攻击模拟**：对模型施加对抗攻击。
*   **解释稳定性**：观察在模型受到攻击时，解释结果（概念归因）的变化情况，评估解释方法的鲁棒性。

### 11. 与 SENN 对比 (`senn`)
*   将本方法的解释结果与自解释神经网络 (Self-Explaining Neural Networks, SENN) 进行对比。

---

## 执行入口
脚本通过 `if __name__ == "__main__":` 块执行，支持命令行参数（如 `--latent_dim`, `--seeds`, `--plot` 等）来控制执行哪些特定的实验环节。默认情况下会依次运行上述大部分实验。
