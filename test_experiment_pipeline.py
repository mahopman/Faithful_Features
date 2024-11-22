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

list_of_questions = ["What is 11356 + 42 + 9863? "]   # Iterable list of questions
instruction_prefix = "Solve the following question: "   # Prefix to be added to all questions
suffix = "Think step by step."   # Suffix prompting the model to show its reasoning
model_output = []


for token in client.chat.completions.create(
    [
        {"role": "user", "content": instruction_prefix + list_of_questions[0] + suffix}
    ],
    model=variant,
    stream=True,
    max_completion_tokens=5000,
):
    #print(token.choices[0].delta.content, end="")
    model_output.append(token.choices[0].delta.content)

assistant_response = ''.join(model_output)


# Create the input structure for feature inspection
context = client.features.inspect(
    [
        {
            "role": "user",
            "content": instruction_prefix + list_of_questions[0] + suffix,
        },
        {
            "role": "assistant",
            "content": assistant_response,
        },
    ],
    model=variant,
)

top_features = context.top(k=10)

#print(top_features)
#print(assistant_response)


# Manually crafted incorrect assistant response
incorrect_reasoning = """
I'd be happy to help you with that!

Let's break it down step by step:

1. First, I'll add 11356 and 42:
11356 + 42 = 11350
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


print(incorrect_response.choices[0].message['content'])


# Feature inspection for incorrect reasoning
incorrect_context = client.features.inspect(
    [
        {
            "role": "user",
            "content": instruction_prefix + list_of_questions[0] + suffix,
        },
        {
            "role": "assistant",
            "content": assistant_response,
        },
    ],
    model=variant,
)

incorrect_top_features = incorrect_context.top(k=10)

print('Correct Top 10 Features')
print(top_features)

print('Incorrect Top 10 Features')
print(incorrect_top_features)

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
                "content": incorrect_reasoning + incorrect_response.choices[0].message['content'],  # Manually crafted incorrect response
            }
        ]
    ],
    dataset_2_feature_rerank_query="incorrect",  # Optional: Focus on a specific trait
    model=variant,
    top_k=10  # Adjust the number of top features as needed
)

print("Features distinguishing incorrect reasoning:")
print(incorrect_features)

"""
# Finding nearest neighbour features for top k different features
for feature in incorrect_features:
    print(feature)
    print(client.features._experimental.neighbors(    # This searches for nearest neighbour features
    feature,
    model=variant
))

"""