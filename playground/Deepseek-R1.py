import json
import requests
import time
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime
import re
from collections import Counter

class ClinicalNoteClassifier:
    def __init__(self, api_key: str, batch_size: int = 5, model: str = None):
        """
        Initialize the classifier with OpenRouter API key
        
        Args:
            api_key (str): OpenRouter API key
            batch_size (int): Number of notes to process in each batch (default: 5)
            model (str): Model to use (default: auto-detect best available)
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Try different model names in order of preference
        self.available_models = [
            "deepseek/deepseek-r1-0528:free",
            "deepseek/deepseek-r1-0528"
        ]
        
        self.model = model if model else self.available_models[0]
        self.batch_size = batch_size
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def calculate_classification_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate classification metrics
        
        Args:
            results (List[Dict]): List of classification results
            
        Returns:
            Dict[str, Any]: metrics including accuracy, precision, recall, F1-score
        """
        if not all('label' in r for r in results):
            return {"error": "Ground truth labels not available for all samples"}
        
        # Extract true and predicted labels
        y_true = [r['label'] for r in results]
        y_pred = [r['predicted_classification'] for r in results if r['predicted_classification'] != 'ERROR']
        
        # Handle cases where some predictions failed
        valid_indices = [i for i, r in enumerate(results) if r['predicted_classification'] != 'ERROR']
        y_true_valid = [y_true[i] for i in valid_indices]
        
        if not y_pred:
            return {"error": "No valid predictions to evaluate"}
        
        # Calculate confusion matrix
        labels = ['IP', 'OP']
        confusion_matrix = {
            'true_positives': {'IP': 0, 'OP': 0},
            'false_positives': {'IP': 0, 'OP': 0},
            'false_negatives': {'IP': 0, 'OP': 0},
            'true_negatives': {'IP': 0, 'OP': 0}
        }
        
        for true_label, pred_label in zip(y_true_valid, y_pred):
            for label in labels:
                if true_label == label and pred_label == label:
                    confusion_matrix['true_positives'][label] += 1
                elif true_label != label and pred_label == label:
                    confusion_matrix['false_positives'][label] += 1
                elif true_label == label and pred_label != label:
                    confusion_matrix['false_negatives'][label] += 1
                else:  # true_label != label and pred_label != label
                    confusion_matrix['true_negatives'][label] += 1
        
        # Calculate metrics for each class
        metrics = {}
        for label in labels:
            tp = confusion_matrix['true_positives'][label]
            fp = confusion_matrix['false_positives'][label]
            fn = confusion_matrix['false_negatives'][label]
            tn = confusion_matrix['true_negatives'][label]
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Support = TP + FN (actual instances of this class)
            support = tp + fn
            
            metrics[label] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1_score, 4),
                'support': support
            }
        
        # Overall accuracy
        correct_predictions = sum(1 for true_label, pred_label in zip(y_true_valid, y_pred) if true_label == pred_label)
        accuracy = correct_predictions / len(y_pred) if y_pred else 0.0
        
        # Macro averages (unweighted average)
        macro_precision = sum(metrics[label]['precision'] for label in labels) / len(labels)
        macro_recall = sum(metrics[label]['recall'] for label in labels) / len(labels)
        macro_f1 = sum(metrics[label]['f1_score'] for label in labels) / len(labels)
        
        # Weighted averages (weighted by support)
        total_support = sum(metrics[label]['support'] for label in labels)
        if total_support > 0:
            weighted_precision = sum(metrics[label]['precision'] * metrics[label]['support'] for label in labels) / total_support
            weighted_recall = sum(metrics[label]['recall'] * metrics[label]['support'] for label in labels) / total_support
            weighted_f1 = sum(metrics[label]['f1_score'] * metrics[label]['support'] for label in labels) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        # Label distribution
        label_distribution = {
            'actual': Counter(y_true),
            'predicted': Counter(y_pred)
        }
        
        # Create confusion matrix in standard format
        confusion_matrix_table = {}
        for true_label in labels:
            confusion_matrix_table[true_label] = {}
            for pred_label in labels:
                count = sum(1 for t, p in zip(y_true_valid, y_pred) if t == true_label and p == pred_label)
                confusion_matrix_table[true_label][pred_label] = count
        
        return {
            'accuracy': round(accuracy, 4),
            'per_class_metrics': metrics,
            'macro_avg': {
                'precision': round(macro_precision, 4),
                'recall': round(macro_recall, 4),
                'f1_score': round(macro_f1, 4),
                'support': total_support
            },
            'weighted_avg': {
                'precision': round(weighted_precision, 4),
                'recall': round(weighted_recall, 4),
                'f1_score': round(weighted_f1, 4),
                'support': total_support
            },
            'confusion_matrix': confusion_matrix_table,
            'label_distribution': label_distribution,
            'total_samples': len(results),
            'valid_predictions': len(y_pred),
            'failed_predictions': len(results) - len(y_pred)
        }

    
    def create_batch_classification_prompt(self, batch_data: List[Dict[str, Any]]) -> str:
    
        """
        Create a prompt for batch processing of clinical notes
        
        Args:
            batch_data (List[Dict]): List of patient data to classify
            
        Returns:
            str: The formatted batch prompt
        """
        prompt = """You are an expert psychiatrist reviewing clinical documentation to make admission decisions.

You will be provided with multiple clinical notes for psychiatric patients. Your task is to assess, based solely on the clinical information in each note, whether each patient requires inpatient (hospital) admission (IP) or can be safely managed as an outpatient (OP).

Instructions:
-Carefully analyze only the clinical facts from each note.
-Do NOT consider explicit discharge or admission instructions unless they are supported by objective clinical findings.
-Base your decision on generally accepted psychiatric admission criteria such as safety, acute risk of harm to self/others, medical/psychiatric instability, inability to care for self, or lack of outpatient supports.
-Do NOT guess or fabricate information not in the notes.
-Select only one answer per patient: "IP" (Inpatient is required) or "OP" (Outpatient is sufficient).
-Provide a concise justification summarizing the main reasons for your decision.
-Highlight key clinical phrases from the note that strongly influenced your decision.

For each patient, provide your response in the following JSON format:
{
  "patient_id": [PATIENT_ID],
  "answer": "IP" or "OP",
  "justification": "Brief explanation of your reasoning",
  "key_phrases": ["phrase1", "phrase2", "phrase3"]
}

Here are the clinical notes to analyze:

"""
        
        for i, patient_data in enumerate(batch_data, 1):
            prompt += f"""
=== PATIENT {patient_data['patient_id']} ===
Clinical Note:
{patient_data['collated_notes']}

"""
        
        prompt += """
Please analyze each patient and provide your assessment in the specified JSON format for each patient."""
        
        return prompt 
    
    def test_model_availability(self) -> bool:
        """
        Test if the current model is available and switch to alternatives if needed
        
        Returns:
            bool: True if a working model is found, False otherwise
        """
        test_payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        for model in self.available_models:
            test_payload["model"] = model
            print(f"Testing model: {model}")
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=test_payload,
                    timeout=30
                )
                print(f"Test response status for {model}: {response.status_code}")
                
                if response.status_code == 200:
                    self.model = model
                    print(f"Using model: {model}")
                    return True
                elif response.status_code == 404:
                    print(f"Model {model} not available (404)")
                    continue
                else:
                    print(f"Model {model} returned status code: {response.status_code}")
                    print(f"Response: {response.text[:200]}")
                    continue
            except Exception as e:
                print(f"Error testing model {model}: {e}")
                continue
        
        print("No working models found!")
        return False
    
    def call_deepseek_model(self, prompt: str) -> Dict[str, Any]:
        """
        Call the DeepSeek reasoning model via OpenRouter
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            Dict[str, Any]: The API response
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.001,  # Low temperature for consistent reasoning (reproducibility)
            "max_tokens": 8000,
        }
        
        try:
            print(f"Making API call to: {self.base_url}")
            print(f"Using model: {self.model}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=120  # Increased timeout
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 404:
                print(f"Model {self.model} not found. Trying alternative models...")
                # Try to find a working model
                if self.test_model_availability():
                    # Retry with the new model
                    payload["model"] = self.model
                    print(f"Retrying with model: {self.model}")
                    response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=120
                    )
                    print(f"Retry response status: {response.status_code}")
                else:
                    print("No working models available!")
                    return None
            
            if response.status_code != 200:
                print(f"Error response content: {response.text[:500]}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text[:500]}")
            return None
    
    def parse_batch_response(self, response_text: str, patient_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Parse batch response and extract JSON responses for each patient
        
        Args:
            response_text (str): The full response from the model
            patient_ids (List[int]): List of patient IDs in the batch
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping patient_id to parsed results
        """
        results = {}
        
        # Try to find JSON objects in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        # Parse each JSON object
        parsed_jsons = []
        for json_match in json_matches:
            try:
                parsed_json = json.loads(json_match)
                if 'patient_id' in parsed_json or 'answer' in parsed_json:
                    parsed_jsons.append(parsed_json)
            except json.JSONDecodeError:
                # Try to clean up common JSON formatting issues
                cleaned_json = json_match.strip()
                # Fix common issues like trailing commas
                cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
                try:
                    parsed_json = json.loads(cleaned_json)
                    if 'patient_id' in parsed_json or 'answer' in parsed_json:
                        parsed_jsons.append(parsed_json)
                except json.JSONDecodeError:
                    continue
        
        # Match JSON responses to patient IDs
        for parsed_json in parsed_jsons:
            patient_id = parsed_json.get('patient_id')
            if patient_id and patient_id in patient_ids:
                answer = parsed_json.get('answer', 'OP')
                justification = parsed_json.get('justification', 'No justification provided')
                key_phrases = parsed_json.get('key_phrases', [])
                
                results[patient_id] = {
                    "raw_reasoning": json.dumps(parsed_json, indent=2),
                    "predicted_classification": answer.upper() if answer.upper() in ['IP', 'OP'] else 'OP',
                    "justification": justification,
                    "key_phrases": key_phrases
                }
        
        # Handle patients not found in JSON responses using fallback parsing
        for patient_id in patient_ids:
            if patient_id not in results:
                # Try to find patient-specific content in the response
                patient_pattern = f"PATIENT {patient_id}"
                if patient_pattern in response_text:
                    # Extract section for this patient
                    start_idx = response_text.find(patient_pattern)
                    next_patient = re.search(r'PATIENT \d+', response_text[start_idx + len(patient_pattern):])
                    end_idx = start_idx + len(patient_pattern) + next_patient.start() if next_patient else len(response_text)
                    
                    section = response_text[start_idx:end_idx]
                    
                    # Try to extract answer from the section
                    answer = "OP"  # Default
                    if re.search(r'"answer":\s*"IP"', section, re.IGNORECASE) or re.search(r'\b(inpatient|IP)\b', section, re.IGNORECASE):
                        answer = "IP"
                    
                    # Try to extract justification
                    justification_match = re.search(r'"justification":\s*"([^"]*)"', section, re.IGNORECASE)
                    justification = justification_match.group(1) if justification_match else "Fallback parsing used"
                    
                    # Try to extract key phrases
                    key_phrases_match = re.search(r'"key_phrases":\s*\[(.*?)\]', section, re.IGNORECASE | re.DOTALL)
                    key_phrases = []
                    if key_phrases_match:
                        phrases_text = key_phrases_match.group(1)
                        key_phrases = re.findall(r'"([^"]*)"', phrases_text)
                    
                    results[patient_id] = {
                        "raw_reasoning": section.strip(),
                        "predicted_classification": answer,
                        "justification": justification,
                        "key_phrases": key_phrases
                    }
                else:
                    # Complete fallback for missing patients
                    results[patient_id] = {
                        "raw_reasoning": f"Patient {patient_id}: Failed to parse response",
                        "predicted_classification": "OP",
                        "justification": "Failed to parse model response",
                        "key_phrases": []
                    }
        
        return results
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of clinical notes
        
        Args:
            batch_data (List[Dict]): Batch of patient data
            
        Returns:
            List[Dict]: Processed results for the batch
        """
        print(f"Processing batch of {len(batch_data)} patients...")
        
        # Create batch prompt
        prompt = self.create_batch_classification_prompt(batch_data)
        
        # Call the model
        response = self.call_deepseek_model(prompt)
        
        if response is None:
            # Handle API failure for entire batch
            results = []
            for patient_data in batch_data:
                result_entry = {
                    **patient_data,
                    "predicted_classification": "ERROR",
                    "raw_reasoning": "Failed to get response from API",
                    "justification": "API call failed",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "processing_timestamp": datetime.now().isoformat()
                }
                results.append(result_entry)
            return results
        
        # Parse batch response
        response_text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        patient_ids = [p['patient_id'] for p in batch_data]
        parsed_results = self.parse_batch_response(response_text, patient_ids)
        
        # Get token counts from usage
        usage = response.get('usage', {})
        total_input_tokens = usage.get('prompt_tokens', 0)
        total_output_tokens = usage.get('completion_tokens', 0)
        
        # Distribute token costs across patients in the batch
        input_tokens_per_patient = total_input_tokens // len(batch_data)
        output_tokens_per_patient = total_output_tokens // len(batch_data)
        
        # Create result entries
        results = []
        for patient_data in batch_data:
            patient_id = patient_data['patient_id']
            parsed_result = parsed_results.get(patient_id, {
                "raw_reasoning": f"Patient {patient_id}: Failed to parse response",
                "predicted_classification": "OP"
            })
            
            result_entry = {
                **patient_data,  # Keep all original fields
                "predicted_classification": parsed_result["predicted_classification"],
                "raw_reasoning": parsed_result["raw_reasoning"],
                "justification": parsed_result.get("justification", parsed_result["raw_reasoning"]),
                "key_phrases": parsed_result.get("key_phrases", []),
                "input_tokens": input_tokens_per_patient,
                "output_tokens": output_tokens_per_patient,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            results.append(result_entry)
        
        return results
    
    def process_clinical_notes(self, input_file: str, output_file: str) -> None:
        """
        Process clinical notes from input file using batch processing
        
        Args:
            input_file (str): Path to input JSON file
            output_file (str): Path to output JSON file
        """
        # Load input data
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Input file {input_file} not found!")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in input file {input_file}")
            return
        
        all_results = []
        total_patients = len(data)
        
        # Process data in batches
        for i in range(0, total_patients, self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_patients + self.batch_size - 1) // self.batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} (patients {i+1}-{min(i+self.batch_size, total_patients)})...")
            
            # Process the batch
            batch_results = self.process_batch(batch_data)
            all_results.extend(batch_results)
            
            # Add delay between batches to avoid rate limiting
            if i + self.batch_size < total_patients:
                print("Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
            
            # Print summary
            ip_predictions = sum(1 for r in all_results if r['predicted_classification'] == 'IP')
            op_predictions = sum(1 for r in all_results if r['predicted_classification'] == 'OP')
            errors = sum(1 for r in all_results if r['predicted_classification'] == 'ERROR')
            
            total_input_tokens = sum(r['input_tokens'] for r in all_results)
            total_output_tokens = sum(r['output_tokens'] for r in all_results)
            
            print(f"\nProcessing Summary:")
            print(f"Total patients: {total_patients}")
            print(f"Predicted Inpatient (IP): {ip_predictions}")
            print(f"Predicted Outpatient (OP): {op_predictions}")
            print(f"Errors: {errors}")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Batch size used: {self.batch_size}")
            print(f"Total API calls: {(total_patients + self.batch_size - 1) // self.batch_size}")
            
            # Calculate accuracy if labels are available
            if all('label' in r for r in all_results):
                print("\nCalculating classification metrics...")
                metrics = self.calculate_classification_metrics(all_results)
                print(metrics)
                # self.print_classification_report(metrics)
                
                # Save metrics to file
                metrics_file = output_file.replace('.json', '_metrics.json')
                try:
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    print(f"\nDetailed metrics saved to: {metrics_file}")
                except Exception as e:
                    print(f"Warning: Could not save metrics file: {e}")
            else:
                print("\nNote: Ground truth labels not available for all samples - cannot calculate classification metrics.")
                
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """
    Main function to run the clinical note classifier with batch processing
    """
    # Configuration
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY=os.getenv('OPENROUTER_KEY')
    if not API_KEY:
        print("Please set the OPENROUTER_API_KEY environment variable")
        API_KEY = input("Enter your OpenRouter API key: ").strip()
    
    INPUT_FILE = "test_graph_in_array_object.json"
    OUTPUT_FILE = "reason_file.json"
    BATCH_SIZE = 10
    
    # Create classifier with batch processing
    classifier = ClinicalNoteClassifier(API_KEY, batch_size=BATCH_SIZE)
    
    # Process the clinical notes
    classifier.process_clinical_notes(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
