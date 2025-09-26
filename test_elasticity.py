#!/usr/bin/env python3
"""
简单测试脚本，验证价格弹性预测功能
"""

import os
import sys
import pandas as pd
from elasticity_prediction import PriceElasticityAnalyzer

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载...")
    analyzer = PriceElasticityAnalyzer()
    
    try:
        analyzer.load_data()
        print("✓ 数据加载成功")
        
        # 检查数据完整性
        assert len(analyzer.zone_data) > 0, "区域数据为空"
        assert len(analyzer.std_devs_data) > 0, "标准差数据为空"
        assert len(analyzer.poi_data) > 0, "POI数据为空"
        print("✓ 数据完整性检查通过")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_filtering():
    """测试数据筛选功能"""
    print("测试数据筛选...")
    analyzer = PriceElasticityAnalyzer()
    
    try:
        analyzer.load_data()
        result = analyzer.filter_dynamic_zones()
        
        if result:
            print("✓ 数据筛选成功")
            print(f"  - 符合条件区域数量: {len(analyzer.filtered_zones)}")
            return True
        else:
            print("✗ 没有符合条件的区域")
            return False
            
    except Exception as e:
        print(f"✗ 数据筛选失败: {e}")
        return False

def test_model_loading():
    """测试模型加载功能"""
    print("测试模型加载...")
    analyzer = PriceElasticityAnalyzer()
    
    try:
        result = analyzer.load_transformer_model()
        if result:
            print("✓ 模型加载成功")
            return True
        else:
            print("✗ 模型加载失败")
            return False
            
    except Exception as e:
        print(f"✗ 模型加载异常: {e}")
        return False

def test_complete_analysis():
    """测试完整分析流程"""
    print("测试完整分析流程...")
    analyzer = PriceElasticityAnalyzer()
    
    try:
        # 运行完整分析（但不显示图表）
        success = analyzer.run_complete_analysis()
        
        if success:
            print("✓ 完整分析流程成功")
            
            # 检查输出文件
            files_generated = []
            for file in os.listdir('.'):
                if file.startswith('elasticity_'):
                    files_generated.append(file)
            
            print(f"  - 生成文件数量: {len(files_generated)}")
            for file in files_generated:
                print(f"    • {file}")
                
            return True
        else:
            print("✗ 完整分析流程失败")
            return False
            
    except Exception as e:
        print(f"✗ 分析过程异常: {e}")
        return False

def main():
    """主测试函数"""
    print("价格弹性预测分析测试")
    print("=" * 40)
    
    tests = [
        test_data_loading,
        test_filtering,
        test_model_loading,
        test_complete_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())