import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
from transformers import AutoTokenizer, AutoModel
import joblib
import spacy
import torch
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


cancer_text = pd.read_csv("data/train_data_01.csv") 
cancer_text = cancer_text.rename(columns={"site": "target"})
procedure_text = cancer_text[["ha_procedure_description", "target"]]
procedure_text = procedure_text.dropna().drop_duplicates()
procedure_text['target_binary'] = procedure_text['target'].apply(lambda x: 0 if x == 'non_cancer' else 1)
X_train= list(procedure_text["ha_procedure_description"])
y_train = procedure_text['target_binary'] 


cancer_text = pd.read_csv("data/test_data_01.csv") 
cancer_text = cancer_text.rename(columns={"site": "target"})
procedure_text = cancer_text[["ha_procedure_description", "target"]]
procedure_text = procedure_text.dropna().drop_duplicates()
procedure_text['target_binary'] = procedure_text['target'].apply(lambda x: 0 if x == 'non_cancer' else 1)
X_test = list(procedure_text["ha_procedure_description"])
y_test = procedure_text['target_binary'] 

nlp = spacy.load("en_core_sci_sm")
device = torch.device("cpu")  # CPU only setup
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)


# Create final pipeline with the best parameters
pipeline = ImbPipeline([
    ('clean', TextCleaner(nlp=nlp)),
    ('embed', BertEmbedder(tokenizer=tokenizer, model=bert_model, device=device)),
    ('pca', PCA(n_components=300)),
    ('under', RandomUnderSampler(random_state=42)),
    ('rf', RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)

# Apply threshold
threshold = 0.4
y_pred = (y_proba[:, 1] >= threshold).astype(int)

print("Recall:", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(pipeline.named_steps['clean'], 'models/text_cleaner.joblib')
joblib.dump(pipeline.named_steps['pca'], 'models/pca_transformer.joblib')
joblib.dump(pipeline.named_steps['rf'], 'models/random_forest_model.joblib')


print("Model and parameters saved.")