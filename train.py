"""
Phishing URL Detection - Advanced Training with LightGBM + Optuna
State-of-the-art character n-gram TF-IDF with hyperparameter optimization
"""

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from lightgbm import LGBMClassifier
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances
)
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("data/phishing_site_urls.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

class Config:
    """Training configuration"""
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    OPTUNA_TRIALS = 80
    OPTUNA_TIMEOUT = 3600  # 1 hour max
    TFIDF_MAX_FEATURES = 1_000_000
    TFIDF_NGRAM_RANGE = (2, 6)
    LGBM_N_ESTIMATORS_TUNING = 4000
    LGBM_N_ESTIMATORS_FINAL = 10000
    EARLY_STOPPING_ROUNDS = 200
    CV_FOLDS = 5

def print_section(title, char="="):
    """Print formatted section header"""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")

def load_and_clean_data():
    """Load and preprocess the phishing dataset"""
    print_section("DATA LOADING", "=")
    
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize column names
    if 'URL' in df.columns and 'Label' in df.columns:
        df = df[['URL', 'Label']].copy()
        df.columns = ['url', 'label']
    elif 'url' in df.columns and 'label' in df.columns:
        df = df[['url', 'label']].copy()
    else:
        raise ValueError(f"Expected columns 'URL'/'Label' or 'url'/'label', got {df.columns.tolist()}")
    
    # Remove missing values
    initial_len = len(df)
    df = df.dropna()
    print(f"Removed {initial_len - len(df)} rows with missing values")
    
    # Standardize labels: 'bad'/'phishing' → 1, 'good'/'legitimate' → 0
    label_mapping = {
        'bad': 1, 
        'good': 0, 
        'phishing': 1, 
        'legitimate': 0,
        1: 1,
        0: 0
    }
    
    df['label'] = df['label'].map(label_mapping)
    
    if df['label'].isna().any():
        print(f"Warning: Found unexpected labels, removing {df['label'].isna().sum()} rows")
        df = df.dropna(subset=['label'])
    
    df['label'] = df['label'].astype(int)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total URLs: {len(df):,}")
    print(f"Phishing URLs: {(df.label == 1).sum():,} ({df.label.mean():.2%})")
    print(f"Legitimate URLs: {(df.label == 0).sum():,} ({(1 - df.label.mean()):.2%})")
    
    return df

def normalize_url(url):
    """
    Normalize URL for feature extraction
    - Remove protocol (http/https)
    - Remove www prefix
    - Convert to lowercase
    """
    if not isinstance(url, str):
        return ""
    
    # Remove protocol and www
    normalized = re.sub(r'^https?://(?:www\.)?', '', url.lower())
    
    # Remove trailing slash
    normalized = normalized.rstrip('/')
    
    return normalized

def create_tfidf_features(df, vectorizer=None, fit=True):
    """
    Create TF-IDF character n-gram features
    
    Character n-grams are excellent for phishing detection because they capture:
    - Typosquatting patterns (gooogle.com vs google.com)
    - Suspicious domain structures
    - Character-level anomalies
    """
    print_section("FEATURE ENGINEERING", "=")
    
    print("Normalizing URLs...")
    df['clean_url'] = df['url'].apply(normalize_url)
    
    # Sample URLs for inspection
    print("\nSample normalized URLs:")
    for i, url in enumerate(df['clean_url'].head(5), 1):
        print(f"   {i}. {url[:80]}...")
    
    if vectorizer is None:
        print(f"\nCreating TF-IDF vectorizer:")
        print(f"   Analyzer: character")
        print(f"   N-gram range: {Config.TFIDF_NGRAM_RANGE}")
        print(f"   Max features: {Config.TFIDF_MAX_FEATURES:,}")
        
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            max_features=Config.TFIDF_MAX_FEATURES,
            dtype='float32',
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True,  # Apply sublinear tf scaling
            min_df=2  # Ignore terms that appear in less than 2 documents
        )
    
    if fit:
        print("\n  Fitting vectorizer and transforming URLs...")
        X = vectorizer.fit_transform(df['clean_url'])
        print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    else:
        print("\n  Transforming URLs...")
        X = vectorizer.transform(df['clean_url'])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])):.4%}")
    print(f"Memory usage: {X.data.nbytes / (1024**2):.1f} MB")
    
    return X, vectorizer

def optuna_objective(trial, X_train, X_test, y_train, y_test):
    """
    Optuna objective function for hyperparameter optimization
    
    This uses Bayesian optimization to find the best hyperparameters
    """
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': Config.RANDOM_STATE,
        
        # Hyperparameters to optimize
        'num_leaves': trial.suggest_int('num_leaves', 128, 1024),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }
    
    model = LGBMClassifier(
        **params,
        n_estimators=Config.LGBM_N_ESTIMATORS_TUNING
    )
    
    # Train with early stopping
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, 'auc'),
        ]
    )
    
    # Predict and calculate AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return auc

def optimize_hyperparameters(X_train, X_test, y_train, y_test):
    """
    Run Optuna hyperparameter optimization
    """
    print_section("HYPERPARAMETER OPTIMIZATION", "=")
    
    print(f"   Starting Optuna study:")
    print(f"   Trials: {Config.OPTUNA_TRIALS}")
    print(f"   Timeout: {Config.OPTUNA_TIMEOUT}s ({Config.OPTUNA_TIMEOUT/60:.0f} minutes)")
    print(f"   Objective: Maximize ROC-AUC")
    print(f"   Sampler: TPE (Tree-structured Parzen Estimator)")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Optimize
    start_time = time.time()
    
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, X_test, y_train, y_test),
        n_trials=Config.OPTUNA_TRIALS,
        timeout=Config.OPTUNA_TIMEOUT,
        show_progress_bar=True,
        n_jobs=1  # LightGBM already uses all cores
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOptimization complete in {elapsed_time/60:.1f} minutes")
    print(f"\n🏆 Best Results:")
    print(f"   Best AUC: {study.best_value:.6f}")
    print(f"   Best trial: #{study.best_trial.number}")
    print(f"\n📊 Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"   {param}: {value}")
    
    # Save study results
    try:
        joblib.dump(study, MODEL_PATH / 'optuna_study.pkl')
        print(f"\nSaved Optuna study to {MODEL_PATH / 'optuna_study.pkl'}")
    except Exception as e:
        print(f"⚠️  Could not save study: {e}")
    
    return study.best_params

def train_final_model(X, y, best_params):
    """
    Train final model on full dataset with optimized hyperparameters
    """
    print_section("FINAL MODEL TRAINING", "=")
    
    print(f"🚀 Training final model with {Config.LGBM_N_ESTIMATORS_FINAL:,} estimators")
    print(f"   Training samples: {X.shape[0]:,}")
    print(f"   Features: {X.shape[1]:,}")
    
    # Add fixed parameters
    final_params = {
        **best_params,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': Config.RANDOM_STATE,
        'n_estimators': Config.LGBM_N_ESTIMATORS_FINAL
    }
    
    model = LGBMClassifier(**final_params)
    
    start_time = time.time()
    model.fit(X, y)
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining complete in {elapsed_time/60:.1f} minutes")
    print(f"Actual number of trees: {model.n_estimators}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    print_section("MODEL EVALUATION", "=")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"📊 Overall Metrics:")
    print(f"   Accuracy: {accuracy:.6f}")
    print(f"   ROC-AUC:  {auc:.6f}")
    print(f"   F1 Score: {f1:.6f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['Legitimate', 'Phishing'],
        digits=4
    ))
    
    print(f"🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                Legit  Phish")
    print(f"   Actual Legit  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"          Phish  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Precision-Recall curve analysis
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Find optimal threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n🎯 Optimal Threshold Analysis:")
    print(f"   Optimal threshold: {optimal_threshold:.4f}")
    print(f"   F1 at optimal: {f1_scores[optimal_idx]:.4f}")
    print(f"   Precision at optimal: {precisions[optimal_idx]:.4f}")
    print(f"   Recall at optimal: {recalls[optimal_idx]:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'optimal_threshold': optimal_threshold
    }

def save_artifacts(model, vectorizer, metadata):
    """
    Save all model artifacts
    """
    print_section("SAVING MODEL ARTIFACTS", "=")
    
    # Save model
    model_file = MODEL_PATH / 'model.pkl'
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")
    
    # Save vectorizer
    vectorizer_file = MODEL_PATH / 'vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_file)
    print(f"Saved vectorizer to {vectorizer_file}")
    
    # Save metadata
    metadata_file = MODEL_PATH / 'metadata.pkl'
    joblib.dump(metadata, metadata_file)
    print(f"Saved metadata to {metadata_file}")
    
    # Print file sizes
    print(f"\n📦 File Sizes:")
    print(f"   Model: {model_file.stat().st_size / (1024**2):.1f} MB")
    print(f"   Vectorizer: {vectorizer_file.stat().st_size / (1024**2):.1f} MB")

def main():
    """
    Main training pipeline
    """
    start_time = time.time()
    
    print_section("PHISHING URL DETECTOR - ADVANCED TRAINING PIPELINE", "#")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    df = load_and_clean_data()
    
    # 2. Create features
    X, vectorizer = create_tfidf_features(df, fit=True)
    y = df['label'].values
    
    # 3. Split data
    print_section("DATA SPLITTING", "=")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        stratify=y, 
        random_state=Config.RANDOM_STATE
    )
    
    print(f"Training set: {X_train.shape[0]:,} samples ({y_train.mean():.2%} phishing)")
    print(f"Test set: {X_test.shape[0]:,} samples ({y_test.mean():.2%} phishing)")
    
    # 4. Optimize hyperparameters
    best_params = optimize_hyperparameters(X_train, X_test, y_train, y_test)
    
    # 5. Train final model
    final_model = train_final_model(X, y, best_params)
    
    # 6. Evaluate on held-out test set
    metrics = evaluate_model(final_model, X_test, y_test)
    
    # 7. Save everything
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'test_size': Config.TEST_SIZE,
        'best_params': best_params,
        'metrics': metrics,
        'tfidf_params': {
            'ngram_range': Config.TFIDF_NGRAM_RANGE,
            'max_features': Config.TFIDF_MAX_FEATURES,
        },
        'config': vars(Config)
    }
    
    save_artifacts(final_model, vectorizer, metadata)
    
    # Final summary
    total_time = time.time() - start_time
    print_section("TRAINING COMPLETE", "#")
    print(f"✅ Total training time: {total_time/60:.1f} minutes")
    print(f"✅ Final AUC: {metrics['auc']:.6f}")
    print(f"✅ Final Accuracy: {metrics['accuracy']:.6f}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Test predictions: python inference.py")
    print(f"   2. Start API: uvicorn api:app --reload")
    print(f"   3. Launch UI: streamlit run app.py")

if __name__ == "__main__":
    main()