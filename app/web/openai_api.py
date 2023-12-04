import os
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def generate_sentence(question, answer):
    input = f'Question: {question}\nAnswer: {answer}'
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentence assistant. Given a question and a one-word answer, you will convert that one-word answer into a complete sentence based on the question without adding any imaginary content."},
            {"role": "user", "content": input}
        ]
    )
    sentence = completion.choices[0].message
    return sentence.content
