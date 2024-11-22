from goodfire import Client
import goodfire
from dotenv import load_dotenv
import os


# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOODFIRE_API_KEY")
client = Client(api_key=api_key)

variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")  # Model variant

list_of_questions = ["Gary starts in a square at the top left of a grid with co-ordinate (0,5). He then moves 3 places to the right, followed by two places down. What co-ordinate does he finish at? "]   # Iterable list of questions
instruction_prefix = "Solve the following question: "   # Prefix to be added to all questions
suffix = "Think step by step."   # Suffix prompting the model to show its reasoning
model_output = []


# Creating the assistant response where it uses its own CoT to answer
correct_response = client.chat.completions.create(
    [
        {"role": "user", "content": instruction_prefix + list_of_questions[0] + suffix}
    ],
    model=variant,
    stream=False,
    max_completion_tokens=5000,
)

assistant_response = correct_response.choices[0].message['content']


# Now creating the incorrect assistant response

# Manually crafted incorrect assistant response
incorrect_reasoning = """
I'd be happy to help you solve this problem.

Let's break it down step by step:

1. Gary starts at the top left of the grid with coordinates (0,5).
2. He then moves 5 places to the right, which means his new x-coordinate is 0 + 5 = 5.
"""

# Prompt the model with the manually provided incorrect reasoning
incorrect_response = client.chat.completions.create(
    [
        {"role": "user", "content": instruction_prefix + list_of_questions[0] + suffix},
        {"role": "assistant", "content": incorrect_reasoning},
    ],
    model=variant,
    stream=False,  
    max_completion_tokens=500,
)


incorrect_assistant_response = incorrect_response.choices[0].message['content']
print(incorrect_reasoning + incorrect_assistant_response)

# Finding the different features present in the incorrect reasoning which aren't in the correct reasoning
_, incorrect_features = client.features.contrast(
    dataset_1=[
        [
            {
                "role": "user",
                "content": instruction_prefix + list_of_questions[0] + suffix
            },
            {
                "role": "assistant",
                "content": assistant_response,  # Original correct response
            }
        ]
    ],
    dataset_2=[
        [
            {
                "role": "user",
                "content": instruction_prefix + list_of_questions[0] + suffix
            },
            {
                "role": "assistant",
                "content": incorrect_reasoning + incorrect_assistant_response,  # Manually crafted incorrect response
            }
        ]
    ],
    #dataset_2_feature_rerank_query="logical reasoning correctness",  # Optional: Focus on a specific trait
    model=variant,
    top_k=10  # Adjust the number of top features as needed
)

print("Features distinguishing incorrect reasoning:")
print(incorrect_features)


# Finding nearest neighbour features for top k different features
for feature in incorrect_features:
    print(feature)
    print(client.features._experimental.neighbors(    # This searches for nearest neighbour features
    feature,
    model=variant
))

