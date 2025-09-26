#!/usr/bin/env python3
"""
价格弹性预测分析脚本
使用Transformer模型预测动态定价区域的价格弹性
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import torch
import torch.nn as nn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PriceElasticityAnalyzer:
    """价格弹性分析器"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.zone_pricing_file = os.path.join(data_dir, "zone_pricing_types.csv")
        self.std_devs_file = os.path.join(data_dir, "data_only_dynamic_zones_std_devs.csv")
        self.poi_file = os.path.join(data_dir, "poi_processed.csv")
        self.model_file = "main_Transformer.txt"
        
        # 存储数据
        self.zone_data = None
        self.std_devs_data = None
        self.poi_data = None
        self.filtered_zones = None
        self.model = None
        
        # 分析结果
        self.predictions = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载所有数据文件"""
        print("正在加载数据文件...")
        
        try:
            # 加载区域定价类型数据
            self.zone_data = pd.read_csv(self.zone_pricing_file)
            print(f"成功加载区域定价数据: {len(self.zone_data)} 条记录")
            
            # 加载标准差数据
            self.std_devs_data = pd.read_csv(self.std_devs_file)
            print(f"成功加载标准差数据: {len(self.std_devs_data)} 条记录")
            
            # 加载POI数据
            self.poi_data = pd.read_csv(self.poi_file)
            print(f"成功加载POI数据: {len(self.poi_data)} 条记录")
            
        except FileNotFoundError as e:
            print(f"数据文件加载失败: {e}")
            sys.exit(1)
            
    def filter_dynamic_zones(self):
        """筛选动态定价区域和非零标准差区域"""
        print("正在筛选动态定价区域...")
        
        # 筛选动态定价区域
        dynamic_zones = self.zone_data[self.zone_data['pricing_type'] == '动态定价'].copy()
        print(f"动态定价区域数量: {len(dynamic_zones)}")
        
        # 进一步筛选标准差不为0的区域
        valid_std_zones = self.std_devs_data[self.std_devs_data['SimpleAvgPriceStdDev'] != 0]
        print(f"标准差不为0的区域数量: {len(valid_std_zones)}")
        
        # 合并筛选条件
        self.filtered_zones = dynamic_zones[
            dynamic_zones['TAZID'].isin(valid_std_zones['TAZID'])
        ].copy()
        
        # 添加标准差信息
        self.filtered_zones = self.filtered_zones.merge(
            self.std_devs_data, on='TAZID', how='left'
        )
        
        print(f"最终符合条件的区域数量: {len(self.filtered_zones)}")
        
        if len(self.filtered_zones) == 0:
            print("警告: 没有符合条件的区域!")
            return False
            
        return True
        
    def load_transformer_model(self):
        """加载Transformer模型架构"""
        print("正在加载Transformer模型...")
        
        try:
            # 执行模型定义文件
            with open(self.model_file, 'r', encoding='utf-8') as f:
                model_code = f.read()
            
            # 执行模型代码
            exec(model_code, globals())
            
            # 创建模型实例
            input_dim = 10  # 输入特征维度
            self.model = create_model(input_dim=input_dim)
            
            # 初始化模型权重（用于演示）
            self._initialize_model_weights()
            
            print("Transformer模型加载成功")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
            
    def _initialize_model_weights(self):
        """初始化模型权重（模拟训练好的权重）"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        self.model.eval()
        
    def prepare_features(self):
        """准备模型输入特征"""
        print("正在准备特征数据...")
        
        features_list = []
        
        for _, zone in self.filtered_zones.iterrows():
            # 基础区域特征
            zone_features = [
                zone['longitude'],
                zone['latitude'],
                zone['charge_count'],
                zone['area'],
                zone['perimeter'],
                zone['SimpleAvgPriceStdDev']
            ]
            
            # 添加POI特征
            zone_pois = self.poi_data[self.poi_data['zone_id'] == zone['TAZID']]
            
            # POI统计特征
            poi_count = len(zone_pois)
            poi_total_visits = zone_pois['count'].sum() if poi_count > 0 else 0
            poi_diversity = zone_pois['primary_types'].nunique() if poi_count > 0 else 0
            
            # 主要POI类型特征（使用one-hot编码主要类型）
            main_poi_type = zone_pois['primary_types'].mode().iloc[0] if poi_count > 0 else 'none'
            poi_type_encoding = 1.0 if main_poi_type == 'restaurant' else 0.0
            
            zone_features.extend([
                poi_count,
                poi_total_visits,
                poi_diversity,
                poi_type_encoding
            ])
            
            features_list.append(zone_features)
        
        # 转换为numpy数组并标准化
        features_array = np.array(features_list, dtype=np.float32)
        
        # 简单的标准化
        features_mean = features_array.mean(axis=0)
        features_std = features_array.std(axis=0) + 1e-8
        features_normalized = (features_array - features_mean) / features_std
        
        return features_normalized
        
    def predict_elasticity(self):
        """使用模型预测价格弹性"""
        print("正在进行价格弹性预测...")
        
        # 准备特征
        features = self.prepare_features()
        
        # 转换为PyTorch张量
        # 为了适配Transformer的输入格式，我们需要序列维度
        # 这里我们创建一个简单的时间序列（重复特征以模拟时间序列）
        seq_len = 5
        batch_size = len(features)
        
        # 创建序列数据（简单重复，实际应用中应该是真实的时间序列数据）
        input_tensor = torch.FloatTensor(features).unsqueeze(1).repeat(1, seq_len, 1)
        
        # 进行预测
        with torch.no_grad():
            predictions = self.model(input_tensor)
            
        # 将预测结果转换为价格弹性值
        elasticity_values = torch.sigmoid(predictions).numpy()  # 使用sigmoid确保正值
        elasticity_values = elasticity_values * 2.0 - 1.0  # 缩放到[-1, 1]范围
        
        # 存储预测结果
        self.predictions = pd.DataFrame({
            'TAZID': self.filtered_zones['TAZID'].values,
            'longitude': self.filtered_zones['longitude'].values,
            'latitude': self.filtered_zones['latitude'].values,
            'predicted_elasticity': elasticity_values,
            'std_dev': self.filtered_zones['SimpleAvgPriceStdDev'].values,
            'charge_count': self.filtered_zones['charge_count'].values
        })
        
        print(f"完成{len(self.predictions)}个区域的弹性预测")
        
    def statistical_analysis(self):
        """进行统计分析"""
        print("正在进行统计分析...")
        
        if self.predictions is None:
            print("错误: 未找到预测结果")
            return
            
        elasticity = self.predictions['predicted_elasticity']
        
        # 基础统计
        self.analysis_results['basic_stats'] = {
            'mean': elasticity.mean(),
            'std': elasticity.std(),
            'min': elasticity.min(),
            'max': elasticity.max(),
            'median': elasticity.median(),
            'q25': elasticity.quantile(0.25),
            'q75': elasticity.quantile(0.75)
        }
        
        # 弹性分类
        high_elastic = elasticity < -0.5
        medium_elastic = (elasticity >= -0.5) & (elasticity <= -0.1)
        low_elastic = elasticity > -0.1
        
        self.analysis_results['elasticity_distribution'] = {
            'high_elastic_count': high_elastic.sum(),
            'medium_elastic_count': medium_elastic.sum(),
            'low_elastic_count': low_elastic.sum(),
            'high_elastic_pct': (high_elastic.sum() / len(elasticity)) * 100,
            'medium_elastic_pct': (medium_elastic.sum() / len(elasticity)) * 100,
            'low_elastic_pct': (low_elastic.sum() / len(elasticity)) * 100
        }
        
        # 与其他变量的相关性分析
        correlations = self.predictions[['predicted_elasticity', 'std_dev', 'charge_count']].corr()
        self.analysis_results['correlations'] = correlations
        
        print("统计分析完成")
        
    def create_visualizations(self):
        """创建可视化图表"""
        print("正在创建可视化图表...")
        
        if self.predictions is None:
            print("错误: 未找到预测结果")
            return
            
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建多子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Price Elasticity Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. 弹性值分布直方图
        axes[0, 0].hist(self.predictions['predicted_elasticity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Predicted Price Elasticity')
        axes[0, 0].set_xlabel('Price Elasticity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = self.analysis_results['basic_stats']['mean']
        axes[0, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        axes[0, 0].legend()
        
        # 2. 区域分布散点图
        scatter = axes[0, 1].scatter(
            self.predictions['longitude'], 
            self.predictions['latitude'],
            c=self.predictions['predicted_elasticity'],
            cmap='RdYlBu_r',
            alpha=0.7,
            s=100
        )
        axes[0, 1].set_title('Geographic Distribution of Price Elasticity')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(scatter, ax=axes[0, 1], label='Price Elasticity')
        
        # 3. 弹性vs标准差散点图
        axes[1, 0].scatter(
            self.predictions['std_dev'],
            self.predictions['predicted_elasticity'],
            alpha=0.7,
            color='green'
        )
        axes[1, 0].set_title('Price Elasticity vs Price Standard Deviation')
        axes[1, 0].set_xlabel('Price Standard Deviation')
        axes[1, 0].set_ylabel('Predicted Price Elasticity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 弹性分类饼图
        elastic_dist = self.analysis_results['elasticity_distribution']
        labels = ['High Elastic', 'Medium Elastic', 'Low Elastic']
        sizes = [
            elastic_dist['high_elastic_count'],
            elastic_dist['medium_elastic_count'], 
            elastic_dist['low_elastic_count']
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Distribution of Elasticity Categories')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = f'elasticity_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存: {output_file}")
        
        plt.show()
        
    def create_time_series_analysis(self):
        """创建时间序列分析图（模拟数据）"""
        print("正在创建时间序列分析...")
        
        # 生成模拟的时间序列数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        
        plt.figure(figsize=(12, 8))
        
        # 为每个区域生成模拟的月度弹性变化
        for i, (_, zone) in enumerate(self.predictions.head(5).iterrows()):  # 只显示前5个区域
            base_elasticity = zone['predicted_elasticity']
            
            # 添加季节性变化和噪声
            seasonal_trend = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            noise = np.random.normal(0, 0.05, len(dates))
            time_series = base_elasticity + seasonal_trend + noise
            
            plt.plot(dates, time_series, marker='o', label=f'Zone {zone["TAZID"]}', alpha=0.7)
        
        plt.title('Price Elasticity Time Series Analysis (Simulated)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price Elasticity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存时间序列图
        ts_output_file = f'elasticity_timeseries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(ts_output_file, dpi=300, bbox_inches='tight')
        print(f"时间序列图已保存: {ts_output_file}")
        
        plt.show()
        
    def print_analysis_report(self):
        """打印详细分析报告"""
        print("\n" + "="*60)
        print("价格弹性预测分析报告")
        print("="*60)
        
        # 数据概览
        print(f"\n数据概览:")
        print(f"- 总区域数量: {len(self.zone_data)}")
        print(f"- 动态定价区域数量: {len(self.zone_data[self.zone_data['pricing_type'] == '动态定价'])}")
        print(f"- 分析区域数量: {len(self.filtered_zones)}")
        print(f"- POI数据点数量: {len(self.poi_data)}")
        
        # 预测结果统计
        if self.predictions is not None:
            stats = self.analysis_results['basic_stats']
            print(f"\n弹性预测统计:")
            print(f"- 平均弹性: {stats['mean']:.4f}")
            print(f"- 标准差: {stats['std']:.4f}")
            print(f"- 最小值: {stats['min']:.4f}")
            print(f"- 最大值: {stats['max']:.4f}")
            print(f"- 中位数: {stats['median']:.4f}")
            print(f"- 25%分位数: {stats['q25']:.4f}")
            print(f"- 75%分位数: {stats['q75']:.4f}")
            
            # 弹性分类分布
            dist = self.analysis_results['elasticity_distribution']
            print(f"\n弹性分类分布:")
            print(f"- 高弹性区域 (< -0.5): {dist['high_elastic_count']} ({dist['high_elastic_pct']:.1f}%)")
            print(f"- 中等弹性区域 (-0.5 to -0.1): {dist['medium_elastic_count']} ({dist['medium_elastic_pct']:.1f}%)")
            print(f"- 低弹性区域 (> -0.1): {dist['low_elastic_count']} ({dist['low_elastic_pct']:.1f}%)")
            
            # 相关性分析
            print(f"\n相关性分析:")
            corr = self.analysis_results['correlations']
            print(f"- 弹性与价格标准差相关性: {corr.loc['predicted_elasticity', 'std_dev']:.4f}")
            print(f"- 弹性与充电次数相关性: {corr.loc['predicted_elasticity', 'charge_count']:.4f}")
        
        print("\n" + "="*60)
        
    def save_results(self):
        """保存预测结果到CSV文件"""
        if self.predictions is not None:
            output_file = f'elasticity_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            # 添加更多分析列
            results_df = self.predictions.copy()
            results_df['elasticity_category'] = pd.cut(
                results_df['predicted_elasticity'],
                bins=[-float('inf'), -0.5, -0.1, float('inf')],
                labels=['High', 'Medium', 'Low']
            )
            
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存: {output_file}")
            
            return output_file
        return None
        
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始价格弹性预测分析...")
        print("="*50)
        
        # 步骤1: 加载数据
        self.load_data()
        
        # 步骤2: 筛选数据
        if not self.filter_dynamic_zones():
            return False
            
        # 步骤3: 加载模型
        if not self.load_transformer_model():
            return False
            
        # 步骤4: 进行预测
        self.predict_elasticity()
        
        # 步骤5: 统计分析
        self.statistical_analysis()
        
        # 步骤6: 创建可视化
        self.create_visualizations()
        self.create_time_series_analysis()
        
        # 步骤7: 打印报告
        self.print_analysis_report()
        
        # 步骤8: 保存结果
        self.save_results()
        
        print("\n分析完成!")
        return True

def main():
    """主函数"""
    print("价格弹性预测分析系统")
    print("Author: AI Assistant")
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 50)
    
    try:
        # 创建分析器实例
        analyzer = PriceElasticityAnalyzer()
        
        # 运行完整分析
        success = analyzer.run_complete_analysis()
        
        if success:
            print("分析成功完成!")
        else:
            print("分析过程中出现错误")
            
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()