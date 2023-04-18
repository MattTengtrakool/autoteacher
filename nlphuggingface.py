from transformers import pipeline
import openai
import json
import re

def generate_questions(text, num_questions=20, max_length=2500):
    truncated_text = truncate_text(text, max_length)
    prompt = f"Generate {num_questions} questions based on the following text:\n{truncated_text}\n"
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use the appropriate engine for GPT-3 or GPT-4
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    questions = re.findall(r"\n(.*?)\?", response.choices[0].text)
    return [q.strip() + "?" for q in questions[:num_questions]]


    
def truncate_text(text, max_length):
    tokens = text.split()
    if len(tokens) > max_length:
        return " ".join(tokens[:max_length])
    return text

with open("client_secret.json", "r") as file:
    client_secret = json.load(file)
    openai.api_key = client_secret["openai_api_key"]

# Read the text_content from the output_nlp.txt file
# Read the text_content from the output_nlp.txt file
with open("output_nlp.txt", "r", encoding="utf-8") as file:
    text_content = file.read()

# Generate prompts
prompts = generate_questions(text_content)

# Set up the question-answering pipeline with a larger model
nlp = pipeline("question-answering", model="bert-large-uncased")

prompts_and_responses = []

for prompt in prompts:
    # Generate a response using the question-answering model
    response = nlp(question=prompt, context=text_content, max_answer_tokens=150, temperature=0.8)
    
    prompts_and_responses.append({
        "prompt": prompt,
        "response": response["answer"]
    })

# Save prompts and responses to a file
with open("prompts_and_responses.txt", "w", encoding="utf-8") as file:
    for item in prompts_and_responses:
        file.write(f"Prompt: {item['prompt']}\nResponse: {item['response']}\n\n")

print("Prompts and responses saved to prompts_and_responses.txt.")
