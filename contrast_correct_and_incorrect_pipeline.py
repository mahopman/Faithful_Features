import pandas as pd
from goodfire import Client
import goodfire
from dotenv import load_dotenv
import os
import pickle

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOODFIRE_API_KEY")

client = Client(api_key=api_key)

variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")

# Questions and incorrect reasoning lists
list_of_questions = [
    "What is 11356 + 42 + 9863?",
    "Gary starts in a square at the top left of a grid with co-ordinate (0,5). He then moves 3 places to the right, followed by two places down. What co-ordinate does he finish at? ",
    "Gary starts in a square at the top left of a grid with co-ordinate (0,5). He then moves 3 places to the right, followed by two places down. What co-ordinate does he finish at? "
]

list_of_incorrect_reasonings = [
    """
    I'd be happy to help you with that!
    Let's break it down step by step:
    1. First, I'll add 11356 and 42:
    11356 + 42 = 11350
    """,
    """
    I'd be happy to help you solve this problem.
    Let's break it down step by step:
    1. Gary starts at the top left of the grid with coordinates (0,5).
    2. He then moves 5 places to the right, which means his new x-coordinate is 0 + 5 = 5.
    """,
    """
    I'd be happy to help you solve this problem.
    Let's break it down step by step:
    1. Gary starts at the top left of the grid with coordinates (0,5).
    2. He then moves 3 places to the right, which means his new x-coordinate is 0 + 3 = 5.
    """
]

# Create output directory if it doesn't exist
output_dir = "results/contrast_experiments"
os.makedirs(output_dir, exist_ok=True)

# Check if progress files exist
experiments_file = f"{output_dir}/experiments_results.csv"
features_file = f"{output_dir}/features_results.csv"
neighbors_file = f"{output_dir}/feature_nearest_neighbors.pkl"

if os.path.exists(experiments_file):
    experiments_df = pd.read_csv(experiments_file)
else:
    experiments_df = pd.DataFrame(columns=['experiment_id', 'user', 'assistant', 'incorrect_reasoning', 'contrast_features'])

if os.path.exists(features_file):
    features_df = pd.read_csv(features_file)
else:
    features_df = pd.DataFrame(columns=['experiment_id', 'feature'])

if os.path.exists(neighbors_file):
    with open(neighbors_file, "rb") as f:
        feature_nearest_neighbors = pickle.load(f)
else:
    feature_nearest_neighbors = {}

# Run the experiment for each question
for i, question in enumerate(list_of_questions):
    print(i)
    experiment_id = f"experiment_{i}"

    # Skip if already processed
    if experiment_id in experiments_df['experiment_id'].values:
        print(f"Skipping already completed experiment {experiment_id}.")
        continue
    
    # Generate correct reasoning response
    prompt = {"role": "user", "content": f"Solve the following question: {question} Think step by step."}
    correct_response = client.chat.completions.create(
        [prompt],
        model=variant,
        stream=False,
        max_completion_tokens=5000
    )
    correct_assistant_response = correct_response.choices[0].message['content']
    
    # Generate incorrect reasoning response
    incorrect_reasoning = list_of_incorrect_reasonings[i]
    incorrect_response = client.chat.completions.create(
        [
            prompt,
            {"role": "assistant", "content": incorrect_reasoning},
        ],
        model=variant,
        stream=False,  
        max_completion_tokens=5000,
    )

    incorrect_assistant_response = incorrect_reasoning + incorrect_response.choices[0].message['content']
    # Generate contrasting features
    _, contrast_features = client.features.contrast(
        dataset_1=[
            [prompt, {"role": "assistant", "content": correct_assistant_response}]
        ],
        dataset_2=[
            [prompt, {"role": "assistant", "content": incorrect_assistant_response}]
        ],
        model=variant,
        top_k=10
    )
    # Save contrast features in experiments DataFrame
    experiments_df = pd.concat([
        experiments_df,
        pd.DataFrame([{
            'experiment_id': experiment_id,
            'question': question,
            'corrrect_assistant_response': correct_assistant_response,
            'incorrect_assistant_response': incorrect_assistant_response,
            'incorrect_reasoning': incorrect_reasoning
        }])
    ])
    
    # Save individual features for easier lookup
    for feature in contrast_features:
        features_df = pd.concat([
            features_df,
            pd.DataFrame([{'experiment_id': experiment_id, 'feature': feature}])
        ])
        
        # Update nearest neighbors dictionary
        if feature not in feature_nearest_neighbors:
            neighbors = client.features._experimental.neighbors(    # This searches for nearest neighbour features
                            feature,
                            model=variant
                        )
            feature_nearest_neighbors[feature] = neighbors

    # Save progress after each iteration
    experiments_df.reset_index(drop=True, inplace=True)
    features_df.reset_index(drop=True, inplace=True)
    
    experiments_df.to_csv(experiments_file, index=False)
    features_df.to_csv(features_file, index=False)
    
    with open(neighbors_file, "wb") as f:
        pickle.dump(feature_nearest_neighbors, f)
    
    print(f"Experiment {experiment_id} completed and saved.")


print("Experiment completed. Results saved to 'experiments_results.csv', 'features_results.csv', and 'feature_nearest_neighbors.pkl'.")
