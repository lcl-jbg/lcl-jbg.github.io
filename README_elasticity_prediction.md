# 价格弹性预测分析脚本

## 概述

本脚本实现了基于Transformer模型的价格弹性预测分析功能，用于分析动态定价区域的充电桩价格弹性。

## 功能特点

- **数据筛选**: 自动筛选动态定价区域和标准差不为零的区域
- **Transformer模型**: 使用深度学习Transformer架构进行弹性预测
- **多维特征**: 结合地理位置、区域属性、POI信息等多维特征
- **可视化分析**: 生成弹性分布图、地理分布图、时间序列分析等
- **统计报告**: 提供详细的统计分析和相关性分析
- **结果导出**: 自动保存预测结果和可视化图表

## 文件结构

```
.
├── data/
│   ├── zone_pricing_types.csv          # 区域定价类型数据
│   ├── data_only_dynamic_zones_std_devs.csv  # 动态区域标准差数据
│   └── poi_processed.csv               # POI数据
├── main_Transformer.txt                # Transformer模型架构
├── elasticity_prediction.py           # 主分析脚本
├── requirements.txt                   # Python依赖包
└── README_elasticity_prediction.md    # 本说明文件
```

## 数据格式

### zone_pricing_types.csv
```csv
TAZID,longitude,latitude,charge_count,area,perimeter,pricing_type
```

### data_only_dynamic_zones_std_devs.csv
```csv
TAZID,SimpleAvgPriceStdDev
```

### poi_processed.csv
```csv
primary_types,longitude,latitude,count,zone_id
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python elasticity_prediction.py
```

## 输出结果

### 1. 控制台输出
- 数据加载进度
- 筛选结果统计
- 模型预测进度
- 详细分析报告

### 2. 可视化图表
- `elasticity_analysis_[timestamp].png`: 弹性分析主图表
- `elasticity_timeseries_[timestamp].png`: 时间序列分析图

### 3. 预测结果文件
- `elasticity_predictions_[timestamp].csv`: 详细预测结果

## 模型架构

使用基于Transformer的深度学习模型：
- **编码器**: 多头注意力机制
- **特征投影**: 线性变换层
- **位置编码**: 正弦余弦位置编码
- **输出层**: 多层感知机回归器

## 分析维度

### 弹性分类
- **高弹性** (< -0.5): 价格敏感度高
- **中等弹性** (-0.5 to -0.1): 价格敏感度中等
- **低弹性** (> -0.1): 价格敏感度低

### 相关性分析
- 弹性与价格标准差的相关性
- 弹性与充电次数的相关性
- 地理位置分布特征

## 技术要求

- Python 3.7+
- PyTorch 1.9.0+
- NumPy, Pandas, Matplotlib, Seaborn
- 内存: 建议8GB以上
- GPU: 可选，用于加速模型推理

## 注意事项

1. 确保数据文件格式正确
2. 动态定价区域数量需大于0
3. 标准差不为0的区域才会被分析
4. 模型权重已预初始化，实际使用需要训练好的权重

## 扩展功能

- 支持自定义模型参数
- 支持批量预测
- 支持多种可视化样式
- 支持导出多种格式结果

## 联系方式

如有问题或建议，请联系开发团队。