import pytest
import pandas as pd
import numpy as np
import pickle
import json
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model_evaluation import (
    load_model,
    load_data,
    evaluate_model,
    save_metrics,
    load_params
)


def test_load_data():
    """Test load_data function with a temporary CSV file"""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("feature1,feature2,label\n")
        f.write("0.1,0.2,0\n")
        f.write("0.3,0.4,1\n")
        f.write("0.5,0.6,0\n")
        temp_path = f.name
    
    try:
        df = load_data(temp_path)
        
        # Verify the dataframe structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['feature1', 'feature2', 'label']
    finally:
        os.unlink(temp_path)


def test_load_model():
    """Test load_model function with a temporary pickle file"""
    # Create a simple model
    model = LogisticRegression()
    X_dummy = np.array([[1, 2], [3, 4], [5, 6]])
    y_dummy = np.array([0, 1, 0])
    model.fit(X_dummy, y_dummy)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(model, f)
        temp_path = f.name
    
    try:
        loaded_model = load_model(temp_path)
        
        # Verify the model is loaded correctly
        assert isinstance(loaded_model, LogisticRegression)
        assert hasattr(loaded_model, 'predict')
    finally:
        os.unlink(temp_path)


def test_evaluate_model():
    """Test evaluate_model function with a trained classifier"""
    # Create and train a simple model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    clf.fit(X_train, y_train)
    
    # Test data
    X_test = np.array([[2, 3], [4, 5]])
    y_test = np.array([0, 1])
    
    # Evaluate the model
    metrics = evaluate_model(clf, X_test, y_test)
    
    # Verify metrics structure
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'auc' in metrics
    
    # Verify metric values are within valid range
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['auc'] <= 1


def test_evaluate_model_perfect_predictions():
    """Test evaluate_model with perfect predictions"""
    # Create a model with perfect predictions
    clf = LogisticRegression()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    clf.fit(X_train, y_train)
    
    # Test with same data (should get perfect score)
    X_test = X_train
    y_test = y_train
    
    metrics = evaluate_model(clf, X_test, y_test)
    
    # With perfect predictions on training data, metrics should be high
    assert metrics['accuracy'] >= 0.9
    assert metrics['precision'] >= 0.9
    assert metrics['recall'] >= 0.9


def test_save_metrics():
    """Test save_metrics function"""
    metrics = {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.88,
        'auc': 0.94
    }
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'reports', 'metrics.json')
        
        # Save metrics
        save_metrics(metrics, file_path)
        
        # Verify file was created
        assert os.path.exists(file_path)
        
        # Load and verify content
        with open(file_path, 'r') as f:
            loaded_metrics = json.load(f)
        
        assert loaded_metrics == metrics
        assert loaded_metrics['accuracy'] == 0.95


def test_load_params_from_yaml():
    """Test load_params function with a temporary YAML file"""
    params_content = """
    model_building:
      n_estimators: 100
      random_state: 42
    feature_engineering:
      max_features: 50
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(params_content)
        temp_path = f.name
    
    try:
        params = load_params(temp_path)
        
        # Verify parameters structure
        assert isinstance(params, dict)
        assert 'model_building' in params
        assert params['model_building']['n_estimators'] == 100
        assert params['model_building']['random_state'] == 42
    finally:
        os.unlink(temp_path)


def test_evaluate_model_with_binary_classification():
    """Test evaluate_model with binary classification scenario"""
    # Create a simple binary classifier
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    
    # Generate more realistic test data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)
    clf.fit(X_train, y_train)
    
    X_test = np.random.rand(30, 5)
    y_test = (X_test[:, 0] + X_test[:, 1] > 1).astype(int)
    
    metrics = evaluate_model(clf, X_test, y_test)
    
    # All metrics should be calculated
    assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'auc'])
    
    # All metrics should be numeric
    assert all(isinstance(metrics[key], (int, float)) for key in metrics)
