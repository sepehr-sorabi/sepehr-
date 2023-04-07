import pandas as pd
import numpy as np

# Define the data
data = {
    ('rf', 'wv_pretrained'): (0.7392722279437006, 0.7412604086615894),
    ('rf', 'wv_custom'): (0.7746309646412634, 0.7762235207436783),
    ('rf', 'glove_pretrained'): (0.7411603158256094, 0.7427841615992615),
    ('rf', 'spacy_pretrained'): (0.731719876416066, 0.7338888018745795),
    ('rf', 'sent_trf'): (0.7229660144181257, 0.7242986991569383),
    ('rf', 'bert_trf'): (0.7126673532440783, 0.7139687043942123),
    ('rf', 'gpt_trf'): (0.7351527634740816, 0.7369294342385289),
    ('rf', 'tfidf'): (0.6920700308959835, 0.6878065672519817),
    ('Logistic_Regression', 'wv_pretrained'): (0.7392722279437006, 0.7412604086615894),
    ('Logistic_Regression', 'wv_custom'): (0.7746309646412634, 0.7762235207436783),
    ('Logistic_Regression', 'glove_pretrained'): (0.7411603158256094, 0.7427841615992615),
    ('Logistic_Regression', 'spacy_pretrained'): (0.731719876416066, 0.7338888018745795),
    ('Logistic_Regression', 'sent_trf'): (0.7229660144181257, 0.7242986991569383),
    ('Logistic_Regression', 'bert_trf'): (0.7126673532440783, 0.7139687043942123),
    ('Logistic_Regression', 'gpt_trf'): (0.7351527634740816, 0.7369294342385289),
    ('Logistic_Regression', 'tfidf'): (0.6920700308959835, 0.6878065672519817)
}

# Create the multi-index
algos = ['rf', 'Logistic_Regression']
embeddings = ['wv_pretrained', 'wv_custom', 'glove_pretrained',
              'spacy_pretrained', 'sent_trf', 'bert_trf', 'gpt_trf', 'tfidf']
eval_metrics = ['accuracy', 'f1_score']
idx = pd.MultiIndex.from_product(
    [algos, embeddings], names=['algos', 'Embedding'])
columns = pd.Index(eval_metrics, name='Metrics')

# Create the dataframe
df = pd.DataFrame(data, index=idx, columns=columns)
df
