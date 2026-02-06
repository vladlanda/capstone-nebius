import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import config

# Import the MixtureOfExpertsRegressor class so joblib can find it
import sys
sys.path.append('.')
from train_mix_of_experts import MixtureOfExpertsRegressor

def analyze_feature_importance(version_name='v1_random', 
                               model_name='xgboost_xgboost',
                               top_n=50):
    """
    Analyze feature importance from trained MoE model.
    
    Args:
        version_name: Data version name
        model_name: Model name (e.g., 'xgboost_xgboost')
        top_n: Number of top features to analyze
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load model
    model_path = Path(config.MODEL_PATH) / f"{version_name}_{model_name}_moe.joblib"
    print(f"\nLoading model from: {model_path}")
    
    try:
        model_pkg = joblib.load(model_path)
        moe_model = model_pkg['model']
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print(f"   Make sure you've trained the model first!")
        return None
    
    # Load training data to get feature names
    X_train_path = f'{config.PROCESSED_DATA_PATH}{version_name}_X_train.csv'
    print(f"Loading features from: {X_train_path}")
    X_train = pd.read_csv(X_train_path)
    
    print(f"\n✓ Loaded {len(X_train.columns)} features")
    
    # Get feature importance from both regressors
    importance_data = {}
    
    if hasattr(moe_model.regressor_high, 'feature_importances_'):
        importance_data['high'] = moe_model.regressor_high.feature_importances_
        print("✓ Got feature importance from HIGH regressor")
    
    if hasattr(moe_model.regressor_low, 'feature_importances_'):
        importance_data['low'] = moe_model.regressor_low.feature_importances_
        print("✓ Got feature importance from LOW regressor")
    
    if not importance_data:
        print("❌ Model doesn't support feature_importances_")
        return None
    
    # Create importance dataframes
    results = {}
    
    for model_type, importances in importance_data.items():
        df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        results[model_type] = df
    
    # Combined importance (average of both)
    if 'high' in results and 'low' in results:
        combined = pd.DataFrame({
            'feature': X_train.columns,
            'importance_high': results['high'].set_index('feature')['importance'],
            'importance_low': results['low'].set_index('feature')['importance']
        })
        combined['importance_avg'] = (combined['importance_high'] + combined['importance_low']) / 2
        combined = combined.sort_values('importance_avg', ascending=False).reset_index(drop=True)
        results['combined'] = combined
    
    # Print analysis
    print("\n" + "="*70)
    print("IMPORTANCE STATISTICS")
    print("="*70)
    
    for model_type, df in results.items():
        if model_type == 'combined':
            imp_col = 'importance_avg'
        else:
            imp_col = 'importance'
        
        print(f"\n{model_type.upper()} Model:")
        print(f"  Total features: {len(df)}")
        
        # Calculate cumulative importance
        cumsum = df[imp_col].cumsum()
        top10_pct = cumsum.iloc[9] / cumsum.iloc[-1] * 100
        top20_pct = cumsum.iloc[19] / cumsum.iloc[-1] * 100
        top30_pct = cumsum.iloc[29] / cumsum.iloc[-1] * 100
        top50_pct = cumsum.iloc[49] / cumsum.iloc[-1] * 100 if len(df) >= 50 else 100
        
        print(f"  Top 10 features: {top10_pct:.1f}% of importance")
        print(f"  Top 20 features: {top20_pct:.1f}% of importance")
        print(f"  Top 30 features: {top30_pct:.1f}% of importance")
        if len(df) >= 50:
            print(f"  Top 50 features: {top50_pct:.1f}% of importance")
        
        print(f"\n  Top 10 features:")
        for i, row in df.head(10).iterrows():
            if model_type == 'combined':
                print(f"    {i+1:2d}. {row['feature']:<40} {row[imp_col]:.6f}")
            else:
                print(f"    {i+1:2d}. {row['feature']:<40} {row[imp_col]:.6f}")
    
    # Save results
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for model_type, df in results.items():
        output_path = results_dir / f"{version_name}_{model_name}_feature_importance_{model_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved: {output_path}")
    
    # Create visualization
    if 'combined' in results:
        plot_feature_importance(results['combined'], version_name, model_name, top_n)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Recommendations based on combined importance
    if 'combined' in results:
        df = results['combined']
        cumsum = df['importance_avg'].cumsum()
        
        # Find how many features for 80%, 90%, 95%
        n_80 = (cumsum / cumsum.iloc[-1] >= 0.80).argmax() + 1
        n_90 = (cumsum / cumsum.iloc[-1] >= 0.90).argmax() + 1
        n_95 = (cumsum / cumsum.iloc[-1] >= 0.95).argmax() + 1
        
        print(f"\n• To retain 80% of importance: Keep top {n_80} features (remove {len(df) - n_80})")
        print(f"• To retain 90% of importance: Keep top {n_90} features (remove {len(df) - n_90})")
        print(f"• To retain 95% of importance: Keep top {n_95} features (remove {len(df) - n_95})")
        
        print(f"\n💡 Recommended: Start by keeping top {n_90} features")
        print(f"   This removes {len(df) - n_90} features while keeping 90% of predictive power")
    
    print("="*70 + "\n")
    
    return results


def plot_feature_importance(df, version_name, model_name, top_n=30):
    """Create visualization of feature importance."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Top N features
    top_features = df.head(top_n)
    ax1 = axes[0]
    sns.barplot(data=top_features, y='feature', x='importance_avg', ax=ax1)
    ax1.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Average Importance', fontsize=12)
    ax1.set_ylabel('Feature', fontsize=12)
    
    # Plot 2: Cumulative importance
    ax2 = axes[1]
    cumsum = df['importance_avg'].cumsum() / df['importance_avg'].sum()
    ax2.plot(range(1, len(cumsum) + 1), cumsum.values, linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% importance')
    ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% importance')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95% importance')
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Cumulative Importance', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path(config.RESULTS_PATH)
    plot_path = results_dir / f"{version_name}_{model_name}_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {plot_path}")
    plt.close()


def create_reduced_dataset(version_name='v1_random', 
                          top_n_features=50,
                          output_version='v1_random_reduced'):
    """
    Create a new dataset with only the top N most important features.
    
    Args:
        version_name: Original data version
        top_n_features: Number of top features to keep
        output_version: Name for the new reduced dataset
    """
    print("\n" + "="*70)
    print(f"CREATING REDUCED DATASET: Top {top_n_features} Features")
    print("="*70)
    
    # Load feature importance
    importance_path = Path(config.RESULTS_PATH) / f"{version_name}_xgboost_xgboost_feature_importance_combined.csv"
    
    if not importance_path.exists():
        print(f"❌ Feature importance file not found: {importance_path}")
        print("   Run analyze_feature_importance() first!")
        return
    
    importance_df = pd.read_csv(importance_path)
    top_features = importance_df.head(top_n_features)['feature'].tolist()
    
    print(f"\n✓ Selected top {len(top_features)} features")
    print(f"\nTop 10 features:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i:2d}. {feat}")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        # Load X
        X_path = f'{config.PROCESSED_DATA_PATH}{version_name}_X_{split}.csv'
        X = pd.read_csv(X_path)
        
        # Select only top features
        X_reduced = X[top_features]
        
        # Save
        output_path = f'{config.PROCESSED_DATA_PATH}{output_version}_X_{split}.csv'
        X_reduced.to_csv(output_path, index=False)
        print(f"✓ Saved {split}: {X_reduced.shape} -> {output_path}")
        
        # Copy y (unchanged)
        y_path = f'{config.PROCESSED_DATA_PATH}{version_name}_y_{split}.csv'
        y_output_path = f'{config.PROCESSED_DATA_PATH}{output_version}_y_{split}.csv'
        y = pd.read_csv(y_path)
        y.to_csv(y_output_path, index=False)
    
    print(f"\n✓ Created reduced dataset: {output_version}")
    print(f"  Original features: {len(X.columns)}")
    print(f"  Reduced features: {len(top_features)}")
    print(f"  Features removed: {len(X.columns) - len(top_features)}")
    
    print("\n💡 Now train on the reduced dataset:")
    print(f"   python train_mix_of_experts.py --version-name {output_version}")
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Selection Analysis')
    parser.add_argument('--version-name', type=str, default='v1_random', 
                       help='Data version to analyze')
    parser.add_argument('--model-name', type=str, default='xgboost_xgboost',
                       help='Model name (e.g., xgboost_xgboost)')
    parser.add_argument('--top-n', type=int, default=30,
                       help='Number of top features to show in plots')
    parser.add_argument('--create-reduced', action='store_true',
                       help='Create reduced dataset with top features')
    parser.add_argument('--keep-features', type=int, default=50,
                       help='Number of features to keep in reduced dataset')
    parser.add_argument('--output-version', type=str, default='v1_random_reduced',
                       help='Output version name for reduced dataset')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_feature_importance(
        version_name=args.version_name,
        model_name=args.model_name,
        top_n=args.top_n
    )
    
    # Optionally create reduced dataset
    if args.create_reduced and results:
        create_reduced_dataset(
            version_name=args.version_name,
            top_n_features=args.keep_features,
            output_version=args.output_version
        )