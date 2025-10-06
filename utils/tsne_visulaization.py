import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from sentence_transformers import SentenceTransformer
import spacy
import re
import string
from empath import Empath

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.like_num or re.search(r'\d', token.text):
            continue
        if token.is_punct or token.text in string.punctuation:
            continue
        if token.ent_type_ == 'PERSON':
            continue
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower()
        tokens.append(lemma)
    return tokens

def extract_trigrams(text):
    tokens = preprocess_text(text)
    return [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

def get_distinctive_trigram_vector(text, label, union_trigrams_list, ip_set, op_set):
    trigrams_in_text = extract_trigrams(text)
    vector = []
    if label.upper() == 'IP':
        for trigram in union_trigrams_list:
            vector.append(trigrams_in_text.count(trigram) if trigram in ip_set else 0)
    elif label.upper() == 'OP':
        for trigram in union_trigrams_list:
            vector.append(trigrams_in_text.count(trigram) if trigram in op_set else 0)
    else:
        vector = [0] * len(union_trigrams_list)
    return np.array(vector, dtype=float)

def get_extended_embedding(text, label, model, lexicon, category_order, union_trigrams_list, ip_set, op_set):
    sentence_emb = model.encode(text)
    impact_features = lexicon.analyze(text, normalize=True)
    impact_feature_vector = np.array([impact_features.get(cat, 0.0) for cat in category_order])
    distinctive_vector = get_distinctive_trigram_vector(text, label, union_trigrams_list, ip_set, op_set)
    extended_embedding = np.concatenate([sentence_emb, impact_feature_vector, distinctive_vector])
    return extended_embedding

def extract_embeddings(graph_data, model, lexicon, category_order, union_trigrams_list, ip_set, op_set):
    embeddings, labels = [], []
    for node in graph_data['nodes']:
        embedding = get_extended_embedding(node['collated_notes'], node['label'], model, lexicon, category_order, union_trigrams_list, ip_set, op_set)
        embeddings.append(embedding)
        labels.append(node['label'])
    return np.array(embeddings), labels

def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    transformed = tsne.fit_transform(embeddings)
    
    label_map = {'IP': 'red', 'OP': 'blue'}
    colors = [label_map[label] for label in labels]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors, alpha=0.6, label=labels)
    
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('t-SNE Visualization of IP and OP Labels')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
               for label, color in label_map.items()]
    plt.legend(handles, label_map.keys())
    plt.show()

if __name__ == "__main__":
    json_file = "/home/user/test_graph.json"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    lexicon = Empath()
    category_order = list(lexicon.cats.keys())
    
    union_trigrams_list = [] 
    ip_set, op_set = set(), set()
    
    graph_data = load_json(json_file)
    embeddings, labels = extract_embeddings(graph_data, model, lexicon, category_order, union_trigrams_list, ip_set, op_set)
    
    plot_tsne(embeddings, labels)
