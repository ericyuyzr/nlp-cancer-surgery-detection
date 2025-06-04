# nlp-cancer-surgery-detection

## Cancer Surgery Classification using BERT and Random Forest

This project uses BERT embeddings and a Random Forest classifier to predict surgery-related prodecures from SPR ha_procedure_description.

---

### 🚀 Project Overview

- Input: Clinical procedure descriptions (`ha_procedure_description`)
- Model: BERT (Bio_ClinicalBERT) + PCA + Random Forest
- Goal: Classify surgeries into binary classes
- Imbalanced data handled using undersampling (during training only)

The SPR database is an important data source, containing surgical records from across British Columbia. However, reviewing new incoming procedures and identifying cancer procedures each month is a manual and time-consuming process.

With the growing power and popularity of large language models (LLMs), we’re exploring whether these tools can help automate this review process and save valuable time.
---

### 🧱 Project Structure

nlp-cancer-surgery-detection/
├── data/ # Input and output CSV files
├── models/ # Saved pipeline and components
├── notebooks/ # Saved data explore, baseline model performance and model selection
├── src/
│ ├── train_pipeline.py # Training script
│ └── inference_model.py # Inference script
├── environment.yml
└── README.md

### 🔬 Approach

### 1. **Baseline Model**
I used facebook/bart-large-mnli with zero shot as the baseline model. If it can perform very well, I don't need to explore other method. The metrics I choosed is recall. Since I don't want to miss any cancer procedures, and we will still review the results given by the model, we could bear the model includs reasonable number of non-cancer procedures. But the result is quite poor. Recall for p

The result of 

### 2. **Text Preprocessing**
- Used **SciSpaCy's `en_core_sci_sm`** model (`spacy.load("en_core_sci_sm")`) to preprocess biomedical text.
- Applied lemmatization, stopword removal, and punctuation filtering.
- This step helped standardize clinical terminology before generating embeddings.

### 📊 Results


### 🚀 Next Steps

- Maybe no need to use **SciSpaCy's `en_core_sci_sm`** .
- Try to train a single layer of classfier instead of using Random Forest. Since we already use the output from BERT, it's natural to still use deep learning airchtechures. 