import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report
import joblib
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


X = np.load("bert_embeddings.npy")  # shape (n_samples, 768)
y = np.load("labels.npy")

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create final pipeline with the best parameters
pipeline = ImbPipeline([
    ('clean', TextCleaner(nlp=nlp)),
    ('embed', BertEmbedder(tokenizer=tokenizer, model=bert_model, device='cuda')),
    ('pca', PCA(n_components=100)),
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

y_pred = pipeline.predict(X_test)
print("Recall:", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save pipeline
joblib.dump(pipeline, "rf_undersample_model.joblib")

best_params = pipeline.named_steps['rf'].get_params()
joblib.dump(best_params, "best_rf_params.joblib")

print("Model and parameters saved.")