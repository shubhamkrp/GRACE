import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os

class TextFeatureExtractor:
    """
    A class to extract various text-based features and combine them into a single
    multi-modal feature vector for each patient record.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initializes the feature extractor with a specified SentenceTransformer model.
        """
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Define keyword lists for clinical domains and risk indicators
        self.clinical_domains = {
            'mood': ['depressive', 'manic', 'bipolar', 'depression', 'mania'],
            'anxiety': ['anxiety', 'panic', 'worry', 'stress'],
            'psychosis': ['psychotic', 'hallucinations', 'delusions', 'insight'],
            'behavioral': ['functioning', 'coping', 'judgment', 'impairment'],
            'social': ['support', 'family', 'isolation', 'therapy']
        }
        self.risk_indicators = {
            'self_harm': ['suicidal', 'self-harm', 'suicide'],
            'functional_impairment': ['impairment', 'unable', 'difficulty'],
            'insight_issues': ['poor insight', 'lack of insight', 'impaired judgment'],
            'support_issues': ['lack of support', 'isolation', 'family issues']
        }

    def _ensure_string(self, text):
        """
        Ensures that the input is converted to a string format.
        Handles cases where text might be a list or other non-string type.
        """
        if isinstance(text, list):
            return ' '.join(str(item) for item in text)
        elif text is None:
            return ""
        else:
            return str(text)

    def extract_justification_embeddings(self, justifications):
        """
        Generates dense vector embeddings for justification texts.
        """
        # Ensure all justifications are strings
        processed_justifications = [self._ensure_string(j) for j in justifications]
        return self.model.encode(processed_justifications, batch_size=8, show_progress_bar=True)

    def extract_clinical_domain_features(self, texts):
        """
        Counts the presence of clinical domain keywords in texts.
        """
        features = np.zeros((len(texts), len(self.clinical_domains)))
        domain_items = list(self.clinical_domains.items())
        for i, text in enumerate(texts):
            # Ensure text is a string before calling .lower()
            text_str = self._ensure_string(text)
            t = text_str.lower()
            for j, (domain, keywords) in enumerate(domain_items):
                features[i, j] = sum(1 for kw in keywords if kw in t)
        return features

    def extract_risk_indicator_features(self, texts):
        """
        Counts the presence of risk indicator keywords in texts.
        """
        features = np.zeros((len(texts), len(self.risk_indicators)))
        risk_items = list(self.risk_indicators.items())
        for i, text in enumerate(texts):
            # Ensure text is a string before calling .lower()
            text_str = self._ensure_string(text)
            t = text_str.lower()
            for j, (risk, keywords) in enumerate(risk_items):
                features[i, j] = sum(1 for kw in keywords if kw in t)
        return features

    def create_multi_modal_features(self, justifications, key_phrases_list):
        """
        Combines all extracted features into a single concatenated vector.
        """
        print("Extracting justification embeddings...")
        text_embed = self.extract_justification_embeddings(justifications)

        print("Extracting clinical domain features...")
        domain_features = self.extract_clinical_domain_features(justifications)

        print("Extracting risk indicator features...")
        risk_features = self.extract_risk_indicator_features(justifications)

        print("Concatenating all features into multi-modal embeddings...")
        multi_modal_features = np.concatenate([
            text_embed, domain_features, risk_features
        ], axis=1)
        
        print(f"Successfully generated multi-modal features with shape: {multi_modal_features.shape}")
        return multi_modal_features

def compute_and_save_embeddings(input_json_path, output_json_path, model_name):
    """
    Main function to orchestrate the loading, processing, and saving of embeddings.
    """
    print(f"Loading patient data from: {input_json_path}")
    with open(input_json_path, 'r') as f:
        results = json.load(f)

    if not isinstance(results, list):
        print("Error: Input JSON file content must be a list of records.")
        return

    patient_ids = [entry.get("patient_id") for entry in results]
    justifications = [entry.get("justification", "") for entry in results]
    key_phrases_list = [entry.get("key_phrases", []) for entry in results]

    # print(f"Total records: {len(justifications)}")
    non_string_count = 0
    for i, j in enumerate(justifications):
        if not isinstance(j, str):
            non_string_count += 1
            if non_string_count <= 3:  # Print first 3 non-string examples
                print(f"Warning: justification at index {i} is {type(j)}: {j}")
    
    if non_string_count > 0:
        print(f"Found {non_string_count} non-string justifications. They will be converted to strings.")

    # Initialize the feature extractor
    feature_extractor = TextFeatureExtractor(model_name=model_name)

    # Generate the multi-modal embeddings
    multi_modal_embeddings = feature_extractor.create_multi_modal_features(
        justifications,
        key_phrases_list
    )

    # Prepare data for JSON output
    output_data = [
        {"patient_id": pid, "embedding": emb.tolist()}
        for pid, emb in zip(patient_ids, multi_modal_embeddings)
    ]

    # Save the results to the output JSON file
    print(f"Saving multi-modal embeddings to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reason embeddings for patient data (clinical note).")
    
    parser.add_argument(
        '--input_json', 
        type=str, 
        required=True, 
        help='Path to the input JSON file. Must contain a "results" list with "patient_id", "justification", and "key_phrases".'
    )
    parser.add_argument(
        '--output_json', 
        type=str, 
        required=True, 
        help='Path to save the output JSON file with patient IDs and their corresponding embeddings.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='sentence-transformers/all-MiniLM-L6-v2', 
        help='Name of the SentenceTransformer model to use.'
    )

    args = parser.parse_args()

    compute_and_save_embeddings(args.input_json, args.output_json, args.model_name)



# python3 reasonEmbeddings.py --input_json input_file.json --output_json reason_embeddings.json
