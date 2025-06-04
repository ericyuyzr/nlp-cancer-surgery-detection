import joblib
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.clean_text(text) for text in X]

    def clean_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                        padding=True, max_length=128).to(self.device)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[0][0].cpu().numpy()
                embeddings.append(cls_embedding)
        return np.array(embeddings)

def predict_on_dataframe(
    df,
    text_column='ha_procedure_description',
    threshold=0.4,
    device='cpu',
    bert_model_name='emilyalsentzer/Bio_ClinicalBERT'
):
    """
    Perform inference on a DataFrame and add prediction columns.

    Args:
        df (pd.DataFrame): Input dataframe with a text column.
        text_column (str): Column name with raw input text.
        threshold (float): Custom threshold for class 1.
        device (str): 'cpu' or 'cuda'.
        bert_model_name (str): Hugging Face model name.

    Returns:
        pd.DataFrame: Original dataframe + predictions + probabilities.
    """

    # Load saved pipeline components
    text_cleaner = joblib.load('models/text_cleaner.joblib')
    pca = joblib.load('models/pca_transformer.joblib')
    model = joblib.load('models/random_forest_model.joblib')

    # Load BERT
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
    bert_model.eval()

    # Clean the text
    raw_texts = df[text_column].astype(str).tolist()
    cleaned_texts = text_cleaner.transform(raw_texts)

    # Generate BERT embeddings
    embeddings = []
    with torch.no_grad():
        for text in cleaned_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            embeddings.append(cls_embedding)

    X_embed = np.array(embeddings)

    # Apply PCA and predict
    X_pca = pca.transform(X_embed)
    y_proba = model.predict_proba(X_pca)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Add results to DataFrame
    df = df.copy()
    df['rf_prediction'] = y_pred
    df['rf_probability'] = y_proba

    return df

df = pd.read_csv("data/test_data_01.csv")

df_with_preds = predict_on_dataframe(df, threshold=0.4, device='cpu')
df_with_preds.to_csv("data/predictions.csv", index=False)