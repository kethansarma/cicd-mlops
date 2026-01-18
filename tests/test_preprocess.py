import pandas as pd
import numpy as np
from src.data_preprocessing import preprocess_df, transform_text
from src.feature_engineering import apply_tfidf


def test_transform_text():
    """Test transform_text function"""
    input_text = "Hello! This is a TEST message."
    result = transform_text(input_text)
    # The result should be lowercase, stemmed, and without stopwords
    assert isinstance(result, str)
    assert result.islower() or result == ""
    assert "!" not in result
    

def test_preprocess_df():
    """Test preprocess_df function"""
    df = pd.DataFrame({
        "text": ["Hello world!", "Test message", "Hello world!"],
        "target": ["ham", "spam", "ham"]
    })
    
    result_df = preprocess_df(df.copy(), text_column='text', target_column='target')
    
    # Check that target is encoded (numeric)
    assert result_df['target'].dtype in [np.int64, np.int32, int]
    
    # Check that duplicates are removed
    assert len(result_df) == 2
    
    # Check that text is transformed
    assert all(result_df['text'].apply(lambda x: isinstance(x, str)))


def test_apply_tfidf():
    """Test apply_tfidf function"""
    train_data = pd.DataFrame({
        "text": ["hello world", "test message", "hello test"],
        "target": [0, 1, 0]
    })
    
    test_data = pd.DataFrame({
        "text": ["hello", "world test"],
        "target": [0, 1]
    })
    
    train_df, test_df = apply_tfidf(train_data, test_data, max_features=10)
    
    # Check that both dataframes have the 'label' column
    assert 'label' in train_df.columns
    assert 'label' in test_df.columns
    
    # Check that the number of rows is preserved
    assert len(train_df) == len(train_data)
    assert len(test_df) == len(test_data)
    
    # Check that labels match the original targets
    assert all(train_df['label'].values == train_data['target'].values)
    assert all(test_df['label'].values == test_data['target'].values)
    
    # Check that TF-IDF features are numeric
    feature_cols = [col for col in train_df.columns if col != 'label']
    assert len(feature_cols) > 0
    assert all(train_df[feature_cols].dtypes.apply(lambda x: np.issubdtype(x, np.number)))