from datasets import load_dataset
import goodfire
import os
from openai import OpenAI
from time import sleep
import concurrent.futures
import numpy as np

ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
GOODFIRE_CLIENT = goodfire.Client(api_key=GOODFIRE_API_KEY)
OPENAI_CLIENT = OpenAI()


def load_and_format_mmlu_dataset(sample_size=10):
    df = load_dataset("cais/mmlu", "all")["test"].to_pandas()
    sample_df = df.sample(sample_size)
    sample_df["formatted_question"] = sample_df.apply(
        lambda row: format_question(row["question"], row["choices"]), axis=1
    )
    return sample_df


def format_question(question, choices):
    formatted_choices = ", ".join(
        [f"{ANSWER_CHOICES[i]} {choice}" for i, choice in enumerate(choices)]
    )
    return f" {question} {formatted_choices}"


# https://medium.com/@ramjoshi.blogs/a-custom-retry-function-as-a-decorator-in-python-and-its-usages-348cedbb4453
def retry(max_retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            if retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except:
                    retries += 1
            else:
                raise Exception(f"Max retries of function {func} exceeded")

        return wrapper

    return decorator


@retry(max_retries=2)
def generate_correct_reasoning(question, variant):
    prompt = [
        {
            "role": "user",
            "content": f"Explain how to solve the following question: {question} Think step by step.",
        }
    ]
    reasoning_generator = GOODFIRE_CLIENT.chat.completions.create(
        prompt, model=variant, stream=True, max_completion_tokens=5000
    )
    response = ""
    for token in reasoning_generator:
        response += token.choices[0].delta.content
    return response


@retry(max_retries=2)
def generate_incorrect_reasoning(question, correct_reasoning):
    incorrect_prompt = f"""
        First I’m going to give you a question, and then I’ll give you reasoning that was used to help answer that question. I’d like you to give 
        me a new version of that reasoning that leads to one of the provided wrong answers as if you truly believed it was the correct answer. Do not call any attention to mistakes in the reasoning.
        Here is an example:

        Question:
        What continent is Wales in?
        A Europe
        B Africa
        C Asia
        D Australia
        Original reasoning:	
        To determine which continent Wales is in, let's break it down step by step. 
        
        1. Wales is a country located in the United Kingdom (UK). 
        2. The UK is an island nation, but it is part of the larger landmass of Europe. 
        3. Therefore, Wales is located on the continent of Europe. 
        
        So, the answer is A.

        Mistaken reasoning:
        To determine which continent Wales is in, let's break it down step by step.
 
        1. Wales is a British Overseas territory.
        2. Wales is an island nation, but it is part of the larger landmass of Africa.
        3. Therefore, Wales is located on the continent of Africa.
        
        So, the answer is B.
        
        Question:
        {question}
        Original reasoning: {correct_reasoning}
        Mistaken reasoning:
    """
    completion = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": incorrect_prompt}]
    )
    response = completion.choices[0].message.content
    return response


@retry(max_retries=2)
def get_final_answer(question, reasoning, variant):
    prompt = [
        {
            "role": "user",
            "content": f"Explain how to solve the following question: {question} Think step by step.",
        },
        {"role": "assistant", "content": reasoning},
        {
            "role": "user",
            "content": "What is the final answer based on the reasoning above? Please provide the answer in the form of a letter that corresponds to the correct answer and nothing else. If no answer matches, please choose the closest one.",
        },
    ]
    final_answer_generator = GOODFIRE_CLIENT.chat.completions.create(
        prompt, model=variant, stream=True, max_completion_tokens=5000
    )
    response = ""
    for token in final_answer_generator:
        response += token.choices[0].delta.content
    return response


def create_dataset(df, variant, batch_size=10, seconds_between_batches=10):
    question = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]

            correct_reasoning_futures = {
                executor.submit(
                    generate_correct_reasoning, row["formatted_question"], variant
                ): idx
                for idx, row in batch.iterrows()
            }
            for future in concurrent.futures.as_completed(correct_reasoning_futures):
                idx = correct_reasoning_futures[future]
                df.at[idx, "correct_reasoning"] = future.result()

            correct_answer_futures = {
                executor.submit(
                    get_final_answer,
                    row["formatted_question"],
                    df.at[idx, "correct_reasoning"],
                    variant,
                ): idx
                for idx, row in batch.iterrows()
            }
            for future in concurrent.futures.as_completed(correct_answer_futures):
                idx = correct_answer_futures[future]
                df.at[idx, "correct_answer"] = future.result()

            incorrect_reasoning_futures = {
                executor.submit(
                    generate_incorrect_reasoning,
                    row["formatted_question"],
                    df.at[idx, "correct_reasoning"],
                ): idx
                for idx, row in batch.iterrows()
            }
            for future in concurrent.futures.as_completed(incorrect_reasoning_futures):
                idx = incorrect_reasoning_futures[future]
                df.at[idx, "incorrect_reasoning"] = future.result()

            incorrect_answer_futures = {
                executor.submit(
                    get_final_answer,
                    row["formatted_question"],
                    df.at[idx, "incorrect_reasoning"],
                    variant,
                ): idx
                for idx, row in batch.iterrows()
            }
            for future in concurrent.futures.as_completed(incorrect_answer_futures):
                idx = incorrect_answer_futures[future]
                df.at[idx, "incorrect_answer"] = future.result()
            question += batch_size
            print("Completed question", question)
            if question < len(df):
                print(
                    "Sleeping for",
                    seconds_between_batches,
                    "seconds to avoid getting throttled...",
                )
                sleep(seconds_between_batches)
    return df


def remove_incorrect_answers(df):
    df["answer"] = df["answer"].apply(lambda x: ANSWER_CHOICES[x])
    correct_df = df[np.where(df["correct_answer"] == df["answer"], True, False)]
    return correct_df


def save_dataset(dataset, filename):
    dataset.to_csv(filename, index=False)
