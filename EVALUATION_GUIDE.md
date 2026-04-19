# LWENet 评估脚本说明

这个项目新增了两类独立评估脚本，均不需要改动现有训练代码：

1. `plot_training_logs.py`
用途：解析 `log/` 目录下的训练日志，生成训练过程曲线和实验汇总表。

2. `evaluate_experiment.py`
用途：加载训练好的 `LWENet` 权重，对指定测试集输出分类指标、ROC 曲线、混淆矩阵和预测结果。

## 1. 训练日志可视化

默认会自动读取 `log/` 下全部 `.log` 文件：

```bash
python plot_training_logs.py
```

也可以手动指定日志：

```bash
python plot_training_logs.py --logs log/wow0.2-train.log
```

默认输出目录：

```text
evaluation_outputs/log_plots/
```

主要产物：

- `*_training_curves.png`
- `*_epoch_metrics.csv`
- `experiment_summary.csv`
- `experiment_comparison.png`

适合写入论文的内容：

- 训练损失曲线
- 验证集/测试集准确率曲线
- 学习率变化曲线
- 不同隐写算法和嵌入率下的精度对比表

## 2. 模型测试集评估

示例：

```bash
python evaluate_experiment.py --weights "checkpoints-wow0.4\lwenet_epoch_200.pkl" --cover-dir "D:\毕业设计\dataset\WOW\0.4bpp\test\cover" --stego-dir "D:\毕业设计\dataset\WOW\0.4bpp\test\stego" --output-dir "evaluation_outputs/wow0.4_eval"
```

默认输出内容：

- `metrics_summary.json`
- `predictions.csv`
- `roc_curve.png`
- `confusion_matrix.png`
- `probability_histogram.png`

终端还会打印以下核心指标：

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Confusion Matrix
- Average Loss

适合写入论文的内容：

- ROC 曲线及 AUC 值
- 混淆矩阵
- Precision / Recall / F1-score
- 概率分布直方图

## 3. 论文中推荐展示的指标

如果你要写毕业论文，建议至少展示下面这些图表和表格：

1. 不同实验的训练损失曲线
2. 不同实验的测试准确率曲线
3. 不同隐写算法/嵌入率的最终准确率对比表
4. 最优模型的 ROC 曲线和 AUC
5. 最优模型的混淆矩阵
6. Precision、Recall、F1-score、Accuracy 汇总表

## 4. 当前仓库内可直接处理的实验

从现有文件看，当前至少可以直接处理：

- `log/suni0.2-train.log`
- `log/suni0.4-train.log`
- `log/wow0.4-train.log`
- `checkpoints-suni0.2/lwenet_epoch_192.pkl`
- `checkpoints-suni0.4/lwenet_epoch_179.pkl`
- `checkpoints-wow0.4/lwenet_epoch_200.pkl`

如果后续你愿意，我下一步可以继续帮你补一层“批量评估入口”，让多组实验一次性生成统一论文图表。
