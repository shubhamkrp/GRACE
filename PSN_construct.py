import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_embeddings(texts, model):
    return model.encode(texts, convert_to_tensor=True).cpu().numpy()

def create_patient_graph(model, input_filepath='/home/user/patient_data.json', output_filepath='/home/user/patient_graph.json'):
    """
    Calculates the cosine similarity
    between the notes of each pair of patients, and creates a graph
    in JSON format with nodes for each patient and edges representing
    the similarity.

    Args:
        input_filepath (str): The path to the input JSON file.
        output_filepath (str): The path to save the output graph JSON file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {input_filepath} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {input_filepath} is not a valid JSON file.")
        return

    # Prepare the data for the graph
    nodes = []
    notes_to_vectorize = []
    patient_ids = []

    for patient in patient_data:
        try:
            patient_id = int(patient['patient_id'])
            collated_notes = patient['collated_notes']
            label = patient['label']

            nodes.append({
                "patient_id": patient_id,
                "collated_notes": collated_notes,
                "label": label
            })
            notes_to_vectorize.append(collated_notes)
            patient_ids.append(patient_id)
        except (KeyError, ValueError) as e:
            print(f"Skipping a record due to missing or invalid data: {e}")
            continue

    if not notes_to_vectorize:
        print("No valid patient notes found to process.")
        return
        
    texts = [node['collated_notes'] for node in nodes]
    embeddings = compute_embeddings(texts, model)

    # Calculate cosine similarity between all pairs of notes
    cosine_sim_matrix = cosine_similarity(embeddings)
    # Create edges for the graph
    edges = []
    num_patients = len(patient_ids)
    for i in range(num_patients):
        for j in range(i + 1, num_patients):
            similarity_score = cosine_sim_matrix[i, j]
            # set a threshold for similarity
            if similarity_score > 0.8:
                edges.append({
                    "source": patient_ids[i],
                    "target": patient_ids[j],
                    "weight": similarity_score
                })

    # Create the final graph structure
    graph = {
        "nodes": nodes,
        "edges": edges
    }

    import numpy as np

    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Clean your graph data
    cleaned_graph = convert_numpy_types(graph)

    with open(output_filepath, 'w') as f:
        json.dump(cleaned_graph, f, indent=2)

    print(f"Graph JSON file created successfully as {output_filepath}")

if __name__ == '__main__':    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    create_patient_graph(model=model)
