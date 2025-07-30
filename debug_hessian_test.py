#!/usr/bin/env python3
"""ヘシアン計算の詳細デバッグテスト"""

import lightgbm as lgb
import numpy as np
import subprocess
import json
import sys
import os
from pathlib import Path

def create_simple_debug_test():
    """シンプルなデバッグテスト用データ"""
    # 非常にシンプルなデータで違いを明確にする
    features = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], 
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    
    return features, labels

def create_debug_rust_program(features, labels):
    """デバッグ用Rustプログラム - ヘシアンと勾配を直接確認"""
    rust_code = f'''//! ヘシアンと勾配のデバッグテスト
use lightgbm_rust::*;
use ndarray::{{Array1, Array2}};
use serde_json;

fn main() -> Result<()> {{
    lightgbm_rust::init()?;
    
    let features = Array2::from_shape_vec(
        ({features.shape[0]}, {features.shape[1]}),
        vec!{features.flatten().tolist()}
    ).expect("特徴量配列の作成に失敗");
    
    let labels = Array1::from_vec(vec!{labels.tolist()});
    
    let dataset = Dataset::new(features.clone(), labels, None, None, None, None)?;
    
    // 1回のイテレーションのみでテスト
    let config = ConfigBuilder::new()
        .objective(ObjectiveType::Binary)
        .num_iterations(1)
        .learning_rate(0.5)  // 大きな学習率で差を明確に
        .num_leaves(3)       // 小さな木で単純化
        .min_data_in_leaf(1)
        .lambda_l2(0.0)      // 正則化なし
        .build()?;
    
    let mut gbdt = GBDT::new(config, dataset)?;
    gbdt.train()?;
    
    let predictions = gbdt.predict(&features)?;
    
    println!("Debug Results:");
    println!("  Features: {{:?}}", features.as_slice().unwrap());
    println!("  Labels: {{:?}}", labels.as_slice().unwrap());
    println!("  Predictions: {{:?}}", predictions);
    
    let result = serde_json::json!({{
        "features": features.as_slice().unwrap(),
        "labels": labels.as_slice().unwrap(),
        "predictions": predictions
    }});
    
    println!("{{}}", serde_json::to_string_pretty(&result).unwrap());
    
    Ok(())
}}'''
    return rust_code

def run_debug_rust_test(rust_code, label):
    """デバッグRustテストを実行"""
    print(f"🦀 {label} でテスト実行中...")
    
    test_file = Path("src/bin/debug_test.rs")
    test_file.parent.mkdir(exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write(rust_code)
    
    try:
        # ビルド
        build_result = subprocess.run(
            ["cargo", "build", "--bin", "debug_test"],
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            print(f"  ❌ ビルドエラー: {build_result.stderr}")
            return None
        
        # 実行
        run_result = subprocess.run(
            ["cargo", "run", "--bin", "debug_test"],
            capture_output=True,
            text=True
        )
        
        if run_result.returncode == 0:
            print(f"  ✅ 実行成功")
            print(f"  出力: {run_result.stdout}")
            
            # JSONパートを抽出
            lines = run_result.stdout.split('\\n')
            json_start = -1
            for i, line in enumerate(lines):
                if line.strip() == '{':
                    json_start = i
                    break
            
            if json_start >= 0:
                json_text = '\\n'.join(lines[json_start:])
                try:
                    result = json.loads(json_text)
                    return result
                except:
                    print(f"  ⚠️ JSON解析失敗: {json_text}")
            
            return run_result.stdout
        else:
            print(f"  ❌ 実行エラー: {run_result.stderr}")
            return None
            
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return None
    finally:
        if test_file.exists():
            test_file.unlink()

def manually_restore_buggy_version():
    """手動でバグ版を復元"""
    objective_path = Path("src/config/objective.rs")
    
    with open(objective_path, 'r') as f:
        content = f.read()
    
    # 現在の修正版を保存
    fixed_version = content
    
    # バグ版に変更（sigmoidを戻す）
    buggy_version = content.replace(
        "let prob = 1.0 / (1.0 + (-predictions[i]).exp());",
        "let prob = 1.0 / (1.0 + (-predictions[i] * self.config.sigmoid).exp());"
    ).replace(
        "hessians[i] = prob * (1.0 - prob);",
        "hessians[i] = prob * (1.0 - prob) * self.config.sigmoid * self.config.sigmoid;"
    )
    
    with open(objective_path, 'w') as f:
        f.write(buggy_version)
    
    print("🔧 バグのあるバージョンに復元しました")
    return fixed_version

def restore_fixed_version(fixed_content):
    """修正版を復元"""
    objective_path = Path("src/config/objective.rs")
    with open(objective_path, 'w') as f:
        f.write(fixed_content)
    print("✅ 修正版に復元しました")

def main():
    """メインデバッグテスト"""
    print("🔍 ヘシアン計算の詳細デバッグテスト")
    print("=" * 50)
    
    # シンプルなテストデータ
    features, labels = create_simple_debug_test()
    print(f"テストデータ:")
    print(f"  特徴量: {features.tolist()}")  
    print(f"  ラベル: {labels.tolist()}")
    
    # Python LightGBMでの基準値
    print("\\n🐍 Python LightGBM基準値:")
    params = {
        'objective': 'binary',
        'num_iterations': 1,
        'learning_rate': 0.5,
        'num_leaves': 3,
        'min_data_in_leaf': 1,
        'lambda_l2': 0.0,
        'verbose': -1,
        'seed': 42,
    }
    
    train_data = lgb.Dataset(features, label=labels)
    model = lgb.train(params, train_data)
    py_pred = model.predict(features)
    print(f"  予測値: {py_pred}")
    
    # Rustプログラム作成
    rust_code = create_debug_rust_program(features, labels)
    
    # 現在は修正版が適用されているので、バグ版をテストするために一時的に変更
    print("\\n" + "="*50)
    fixed_content = manually_restore_buggy_version()
    
    # バグ版でテスト
    print("\\n🦀 バグ版でテスト:")
    buggy_result = run_debug_rust_test(rust_code, "バグ版")
    
    # 修正版に戻す  
    print("\\n" + "="*50)
    restore_fixed_version(fixed_content)
    
    # 修正版でテスト
    print("\\n🦀 修正版でテスト:")
    fixed_result = run_debug_rust_test(rust_code, "修正版")
    
    # 結果比較
    print("\\n" + "="*50)
    print("📊 **詳細比較結果**")
    
    if buggy_result and fixed_result:
        if isinstance(buggy_result, dict) and isinstance(fixed_result, dict):
            buggy_pred = buggy_result.get('predictions', {}).get('data', [])
            fixed_pred = fixed_result.get('predictions', {}).get('data', [])
            
            print(f"Python LightGBM: {py_pred}")
            print(f"Rust バグ版:     {buggy_pred}")
            print(f"Rust 修正版:     {fixed_pred}")
            
            if buggy_pred and fixed_pred:
                buggy_diff = np.abs(np.array(py_pred) - np.array(buggy_pred))
                fixed_diff = np.abs(np.array(py_pred) - np.array(fixed_pred))
                
                print(f"\\nPython基準との差異:")
                print(f"  バグ版:  最大 {np.max(buggy_diff):.6e}, 平均 {np.mean(buggy_diff):.6e}")
                print(f"  修正版:  最大 {np.max(fixed_diff):.6e}, 平均 {np.mean(fixed_diff):.6e}")
                
                if np.max(fixed_diff) < np.max(buggy_diff):
                    print("  ✅ 修正版の方が良い結果!")
                elif np.max(fixed_diff) > np.max(buggy_diff):
                    print("  ❌ バグ版の方が良い結果...")
                else:
                    print("  ⚠️ 差異なし")
        else:
            print("結果の形式が予期しないものでした")
            print(f"バグ版結果: {buggy_result}")
            print(f"修正版結果: {fixed_result}")
    else:
        print("❌ テストの実行に失敗しました")

if __name__ == "__main__":
    main()