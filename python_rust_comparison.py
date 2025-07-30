#!/usr/bin/env python3
"""Python LightGBMとRust LightGBMの出力値差分の詳細分析"""

import lightgbm as lgb
import numpy as np
import subprocess
import json
import sys
import os
from pathlib import Path

def create_comprehensive_test_datasets():
    """包括的なテストデータセットを作成"""
    datasets = {}
    
    # Dataset 1: 基本的なバイナリ分類データ
    np.random.seed(42)
    features1 = np.array([
        [1.0, 2.0, 0.5], [2.0, 1.0, 1.0], [3.0, 0.5, 2.0], [0.5, 3.0, 0.2],
        [2.5, 1.5, 1.2], [1.5, 2.5, 0.8], [3.2, 0.8, 1.8], [0.8, 2.8, 0.5],
        [2.0, 2.0, 1.0], [1.0, 1.0, 1.5], [3.5, 1.2, 2.2], [1.2, 3.2, 0.3],
        [2.8, 1.8, 1.5], [1.8, 2.2, 1.1], [3.0, 1.0, 2.0], [1.0, 3.0, 0.8]
    ], dtype=np.float32)
    labels1 = ((features1[:, 0] + features1[:, 1] - features1[:, 2]) > 2.0).astype(np.float32)
    datasets['binary_basic'] = (features1, labels1, 'binary')
    
    # Dataset 2: 回帰データ
    features2 = np.array([
        [1.0, 2.0, 0.5], [2.0, 1.0, 1.0], [3.0, 0.5, 2.0], [0.5, 3.0, 0.2],
        [2.5, 1.5, 1.2], [1.5, 2.5, 0.8], [3.2, 0.8, 1.8], [0.8, 2.8, 0.5],
        [2.0, 2.0, 1.0], [1.0, 1.0, 1.5]
    ], dtype=np.float32)
    labels2 = np.array([
        2.0 * features2[i, 0] + 3.0 * features2[i, 1] - features2[i, 2] + (i * 0.1)
        for i in range(len(features2))
    ], dtype=np.float32)
    datasets['regression'] = (features2, labels2, 'regression')
    
    # Dataset 3: 異なるスケールのバイナリ分類
    features3 = np.array([
        [10.0, 20.0, 5.0], [20.0, 10.0, 10.0], [30.0, 5.0, 20.0], [5.0, 30.0, 2.0],
        [25.0, 15.0, 12.0], [15.0, 25.0, 8.0], [32.0, 8.0, 18.0], [8.0, 28.0, 5.0]
    ], dtype=np.float32)
    labels3 = ((features3[:, 0] + features3[:, 1] - features3[:, 2]) > 30.0).astype(np.float32)
    datasets['binary_scaled'] = (features3, labels3, 'binary')
    
    return datasets

def train_python_lightgbm(features, labels, objective_type):
    """Python LightGBMでモデルを訓練"""
    if objective_type == 'binary':
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_iterations': 10,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_data_in_leaf': 1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42,
        }
    else:  # regression
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'num_iterations': 10,
            'learning_rate': 0.1,
            'num_leaves': 7,
            'min_data_in_leaf': 1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'seed': 42,
        }
    
    train_data = lgb.Dataset(features, label=labels)
    model = lgb.train(params, train_data)
    predictions = model.predict(features)
    
    return predictions

def create_rust_test_program(datasets):
    """Rustテストプログラムを作成"""
    rust_code = '''//! Python vs Rust LightGBM比較テスト
use lightgbm_rust::*;
use ndarray::{Array1, Array2};
use serde_json;
use std::collections::HashMap;

fn main() -> Result<()> {
    lightgbm_rust::init()?;
    
    let mut results = HashMap::new();
    
'''
    
    for dataset_name, (features, labels, objective_type) in datasets.items():
        if objective_type == 'binary':
            objective_enum = 'ObjectiveType::Binary'
        else:
            objective_enum = 'ObjectiveType::Regression'
            
        rust_code += f'''
    // {dataset_name} dataset
    let features_{dataset_name} = Array2::from_shape_vec(
        ({features.shape[0]}, {features.shape[1]}),
        vec!{features.flatten().tolist()}
    ).expect("特徴量配列の作成に失敗");
    
    let labels_{dataset_name} = Array1::from_vec(vec!{labels.tolist()});
    
    let dataset_{dataset_name} = Dataset::new(
        features_{dataset_name}.clone(), 
        labels_{dataset_name}, 
        None, None, None, None
    )?;
    
    let config_{dataset_name} = ConfigBuilder::new()
        .objective({objective_enum})
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(7)
        .min_data_in_leaf(1)
        .lambda_l2(0.1)
        .build()?;
    
    let mut gbdt_{dataset_name} = GBDT::new(config_{dataset_name}, dataset_{dataset_name})?;
    gbdt_{dataset_name}.train()?;
    
    let predictions_{dataset_name} = gbdt_{dataset_name}.predict(&features_{dataset_name})?;
    
    results.insert("{dataset_name}", serde_json::json!({{
        "predictions": predictions_{dataset_name},
        "objective": "{objective_type}"
    }}));
'''
    
    rust_code += '''
    
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
    
    Ok(())
}'''
    
    return rust_code

def run_rust_test(rust_code):
    """Rustテストを実行"""
    test_file = Path("src/bin/comparison_full.rs")
    
    try:
        test_file.parent.mkdir(exist_ok=True)
        
        with open(test_file, 'w') as f:
            f.write(rust_code)
        
        # ビルド
        build_result = subprocess.run(
            ["cargo", "build", "--bin", "comparison_full"],
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            print(f"❌ ビルドエラー: {build_result.stderr}")
            return None
        
        # 実行
        run_result = subprocess.run(
            ["cargo", "run", "--bin", "comparison_full"],
            capture_output=True,
            text=True
        )
        
        if run_result.returncode == 0:
            try:
                return json.loads(run_result.stdout)
            except json.JSONDecodeError:
                print(f"❌ JSON解析失敗: {run_result.stdout}")
                return None
        else:
            print(f"❌ 実行エラー: {run_result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None
    finally:
        if test_file.exists():
            test_file.unlink()

def analyze_differences(py_pred, rust_pred, dataset_name, objective_type):
    """予測値の差異を詳細分析"""
    py_array = np.array(py_pred)
    
    if isinstance(rust_pred, dict) and 'data' in rust_pred:
        rust_array = np.array(rust_pred['data'])
    else:
        rust_array = np.array(rust_pred)
    
    # 基本統計
    abs_diff = np.abs(py_array - rust_array)
    rel_diff = np.abs((py_array - rust_array) / (py_array + 1e-10))  # 相対誤差（ゼロ除算回避）
    
    stats = {
        'max_absolute_diff': float(np.max(abs_diff)),
        'mean_absolute_diff': float(np.mean(abs_diff)),
        'median_absolute_diff': float(np.median(abs_diff)),
        'std_absolute_diff': float(np.std(abs_diff)),
        'max_relative_diff': float(np.max(rel_diff)) * 100,  # パーセント
        'mean_relative_diff': float(np.mean(rel_diff)) * 100,
        'rmse': float(np.sqrt(np.mean((py_array - rust_array) ** 2))),
        'python_range': [float(np.min(py_array)), float(np.max(py_array))],
        'rust_range': [float(np.min(rust_array)), float(np.max(rust_array))],
        'correlation': float(np.corrcoef(py_array, rust_array)[0, 1]) if len(py_array) > 1 else 1.0
    }
    
    return stats

def main():
    """メイン比較分析"""
    print("🔍 Python LightGBM vs Rust LightGBM 詳細差分分析")
    print("=" * 60)
    
    # テストデータセット作成
    datasets = create_comprehensive_test_datasets()
    print(f"📊 {len(datasets)}個のデータセットで比較テスト")
    
    # Python LightGBMで全データセットを実行
    print("\n🐍 Python LightGBMで予測実行中...")
    python_results = {}
    for dataset_name, (features, labels, objective_type) in datasets.items():
        print(f"  - {dataset_name} ({objective_type})")
        python_results[dataset_name] = {
            'predictions': train_python_lightgbm(features, labels, objective_type),
            'objective': objective_type
        }
    
    # Rustテストプログラム作成・実行
    print("\n🦀 Rust LightGBMで予測実行中...")
    rust_code = create_rust_test_program(datasets)
    rust_results = run_rust_test(rust_code)
    
    if not rust_results:
        print("❌ Rustテストの実行に失敗しました")
        return
    
    # 詳細分析
    print("\n📈 **詳細差分分析結果**")
    print("=" * 60)
    
    overall_stats = {
        'max_diffs': [],
        'mean_diffs': [],
        'correlations': []
    }
    
    for dataset_name in datasets.keys():
        print(f"\n🎯 **{dataset_name.upper()}** データセット:")
        
        py_pred = python_results[dataset_name]['predictions']
        rust_pred = rust_results[dataset_name]['predictions']
        objective_type = python_results[dataset_name]['objective']
        
        stats = analyze_differences(py_pred, rust_pred, dataset_name, objective_type)
        
        print(f"  📊 基本統計:")
        print(f"    最大絶対差:     {stats['max_absolute_diff']:.8e}")
        print(f"    平均絶対差:     {stats['mean_absolute_diff']:.8e}")
        print(f"    中央値絶対差:   {stats['median_absolute_diff']:.8e}")
        print(f"    標準偏差:       {stats['std_absolute_diff']:.8e}")
        print(f"    RMSE:          {stats['rmse']:.8e}")
        
        print(f"  📈 相対誤差:")
        print(f"    最大相対差:     {stats['max_relative_diff']:.6f}%")
        print(f"    平均相対差:     {stats['mean_relative_diff']:.6f}%")
        
        print(f"  🎯 予測値範囲:")
        print(f"    Python:        [{stats['python_range'][0]:.6f}, {stats['python_range'][1]:.6f}]")
        print(f"    Rust:          [{stats['rust_range'][0]:.6f}, {stats['rust_range'][1]:.6f}]")
        print(f"    相関係数:       {stats['correlation']:.8f}")
        
        # 精度評価
        if stats['max_absolute_diff'] < 1e-10:
            precision_level = "🎉 極めて高精度 (< 1e-10)"
        elif stats['max_absolute_diff'] < 1e-8:
            precision_level = "✅ 非常に高精度 (< 1e-8)"
        elif stats['max_absolute_diff'] < 1e-6:
            precision_level = "✅ 高精度 (< 1e-6)"
        elif stats['max_absolute_diff'] < 1e-4:
            precision_level = "⚠️  中程度の精度 (< 1e-4)"
        else:
            precision_level = "❌ 低精度 (>= 1e-4)"
        
        print(f"  🏆 精度評価:      {precision_level}")
        
        # 統計収集
        overall_stats['max_diffs'].append(stats['max_absolute_diff'])
        overall_stats['mean_diffs'].append(stats['mean_absolute_diff'])
        overall_stats['correlations'].append(stats['correlation'])
    
    # 全体総括
    print(f"\n🏆 **全体総括**")
    print(f"=" * 60)
    print(f"全データセット中の:")
    print(f"  最大の最大絶対差:   {max(overall_stats['max_diffs']):.8e}")
    print(f"  最小の最大絶対差:   {min(overall_stats['max_diffs']):.8e}")
    print(f"  平均の最大絶対差:   {np.mean(overall_stats['max_diffs']):.8e}")
    print(f"  平均相関係数:       {np.mean(overall_stats['correlations']):.8f}")
    
    # 最終評価
    max_overall_diff = max(overall_stats['max_diffs'])
    min_correlation = min(overall_stats['correlations'])
    
    if max_overall_diff < 1e-8 and min_correlation > 0.999:
        final_grade = "🎉 **優秀** - Python LightGBMとほぼ同等の精度"
    elif max_overall_diff < 1e-6 and min_correlation > 0.99:
        final_grade = "✅ **良好** - 実用的な精度レベル"
    elif max_overall_diff < 1e-4:
        final_grade = "⚠️  **改善の余地あり** - 中程度の差異"
    else:
        final_grade = "❌ **要改善** - 大きな差異あり"
    
    print(f"\n🎯 **最終評価**: {final_grade}")
    
    return max_overall_diff

if __name__ == "__main__":
    max_diff = main()
    if max_diff < 1e-6:
        print(f"\n✅ **結論**: Rust実装は高精度 (最大差異 {max_diff:.2e})")
    else:
        print(f"\n⚠️  **結論**: 改善の余地あり (最大差異 {max_diff:.2e})")