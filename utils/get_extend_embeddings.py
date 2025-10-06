import numpy as np
from sentence_transformers import SentenceTransformer
from empath import Empath

def get_extended_embedding(text, label, model, lexicon, category_order, union_trigrams_list, ip_set, op_set):
    """
    Compute the extended embedding by concatenating:
      1. The SentenceTransformer sentence embedding.
      2. The normalized Empath lexical features (ordered by category_order).
      3. The distinctive trigram frequency vector (of fixed dimension from the union of significant trigrams).
    """
    # Sentence transformer embedding
    sentence_emb = model.encode(text)
    
    # Empath features in a fixed sorted order
    impact_features = lexicon.analyze(text, normalize=True)
    impact_feature_vector = np.array([impact_features.get(cat, 0.0) for cat in category_order])
    
    # Distinctive trigram features (fixed dimension for all notes)
    distinctive_vector = get_distinctive_trigram_vector(text, label, union_trigrams_list, ip_set, op_set)
    
    # Concatenate all features
    extended_embedding = np.concatenate([sentence_emb, impact_feature_vector, distinctive_vector])
    return extended_embedding
