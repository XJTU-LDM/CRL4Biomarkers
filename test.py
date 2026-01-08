# -*- coding: utf-8 -*-
"""
同步版特征组合logrank检验计算程序
Created on Wed Apr  9 22:44:34 2025
@author: wuli
"""

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
import os
import warnings
warnings.filterwarnings('ignore')

# ================== 配置参数 ==================
DATA_PATHS = {
    "Ntest": "Ntest.xlsx",
    "Ntrain": "Ntrain.xlsx", 
    "Nvalidate": "Nvalidate.xlsx",
    "test": "训练组.xlsx",
    "train": "测试组.xlsx",
    "validate": "验证组整体.xlsx"
}

FEATURE_CSV = "selected_features_train1+2.csv"
OUTPUT_CSV = "all_datasets_pvalues.csv"

# ================== 数据预加载 ==================
def load_all_datasets():
    """主进程一次性加载所有数据"""
    print("📚 主进程加载数据...")
    datasets = {}
    for name, path in DATA_PATHS.items():
        df = pd.read_excel(path)
        time_col = df.iloc[:, 0].values.astype(np.float32)
        event_col = df.iloc[:, 1].values
        features = df.iloc[:, 2:].values.astype(np.float32)
        datasets[name] = (time_col, event_col, features)
    return datasets

# ================== p值计算函数 ==================
def calculate_pvalue_sync(args, datasets):
    """同步计算单个数据集的p值"""
    feature_indices, dataset_name = args
    
    try:
        time_data, event_data, features = datasets[dataset_name]
        
        # 创建特征选择向量
        action = np.zeros(features.shape[1], dtype=int)
        action[feature_indices] = 1
        
        selected = (features @ action) > 0.5
        group1_samples = np.sum(selected)
        group2_samples = np.sum(~selected)
        
        if group1_samples < 3 or group2_samples < 3:
            return (dataset_name, 1.0, False, group1_samples, group2_samples)
        
        result = logrank_test(time_data[selected], 
                            time_data[~selected],
                            event_observed_A=event_data[selected],
                            event_observed_B=event_data[~selected])
        return (dataset_name, result.p_value, True, group1_samples, group2_samples)
    except Exception as e:
        print(f"⚠️ 计算错误: {str(e)}")
        return (dataset_name, 1.0, False, 0, 0)

# ================== 主处理流程 ==================
def main():
    # 加载数据
    all_datasets = load_all_datasets()
    
    # 加载特征组合
    print("📖 读取特征组合文件...")
    features_df = pd.read_csv(FEATURE_CSV)
    features_df['feature_indices'] = features_df['features'].apply(
        lambda x: list(map(int, x.split(','))) if pd.notnull(x) else []
    )
    
    # 生成任务参数
    print("⚙️ 生成计算任务...")
    task_args = []
    for _, row in features_df.iterrows():
        indices = row['feature_indices']
        for ds_name in DATA_PATHS.keys():
            task_args.append( (indices, ds_name) )
    
    # 同步计算
    print("⚡ 开始同步计算...")
    results = []
    
    try:
        from tqdm import tqdm
        # 使用tqdm显示进度条
        with tqdm(total=len(task_args), desc="处理进度") as pbar:
            for args in task_args:
                res = calculate_pvalue_sync(args, all_datasets)
                results.append(res)
                pbar.update()
    except ImportError:
        # 无tqdm时的回退方案
        for i, args in enumerate(task_args):
            res = calculate_pvalue_sync(args, all_datasets)
            results.append(res)
            if i % 100 == 0:
                print(f"已处理 {i}/{len(task_args)} 个任务")
    
    # 构建结果字典（与原逻辑一致）
    print("📊 整理结果...")
    result_map = {}
    for idx, (ds_name, p, valid, n1, n2) in enumerate(results):
        task_idx = idx // len(DATA_PATHS)
        if task_idx not in result_map:
            result_map[task_idx] = {}
        result_map[task_idx][ds_name] = (p, valid, n1, n2)
    
    # 构建输出DataFrame（与原逻辑一致）
    output_data = []
    for idx, row in features_df.iterrows():
        record = {
            "episode": row["episode"],
            "features": row["features"],
            "num_features": row["num_features"],
            "original_reward": row["reward"]
        }
        
        if idx in result_map:
            for ds_name in DATA_PATHS.keys():
                p, valid, n1, n2 = result_map[idx].get(ds_name, (1.0, False, 0, 0))
                record.update({
                    f"{ds_name}_p": p,
                    f"{ds_name}_valid": valid,
                    f"{ds_name}_n1": n1,
                    f"{ds_name}_n2": n2
                })
        
        output_data.append(record)
    
    # 保存结果
    output_df = pd.DataFrame(output_data)
    columns_order = ["episode", "num_features", "original_reward", "features"] 
    for ds in DATA_PATHS.keys():
        columns_order.extend([f"{ds}_p", f"{ds}_valid", f"{ds}_n1", f"{ds}_n2"])
    
    output_df = output_df[columns_order]
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 结果已保存至：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()