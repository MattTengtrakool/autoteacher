import os
import json
import re
from transformers import pipeline,GPT2Tokenizer
from google.cloud import language_v1
import openai

# Set the environment variable for Google API authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/matt/downloads/vaulted-bus-383605-bcb39e1c743c.json"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
with open("client_secret.json", "r") as file:
    client_secret = json.load(file)
    openai.api_key = client_secret["openai_api_key"]

def analyze_entities(text_content):
    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    document = language_v1.Document(content=text_content, type_=type_, language="en")

    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entities(request={'document': document, 'encoding_type': encoding_type})

    return response

def get_openai_answer(prompt, context):
    max_context_tokens = 4096 - 150 - 1  # Reserve 150 tokens for the completion and 1 for the separator
    
    context_tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
    truncated_context = tokenizer.decode(context_tokens)
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{prompt} {truncated_context}",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

input_folder = "output_nlp"
output_file = "prompts_and_responsesnew.txt"

prompts_and_responses = []

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    print(f"Processing {filename}...")
    file_path = os.path.join(input_folder, filename)

    # Read the text_content from the file
    with open(file_path, "r", encoding="utf-8") as file:
        text_content = file.read()

    # Analyze the text_content
    response = analyze_entities(text_content)

    for entity in response.entities:
        # Ask OpenAI API to generate a question related to the entity
        question_prompt = f"Generate a question about {entity.name}:"
        question = get_openai_answer(question_prompt, text_content)
        print(f"Generated question: {question}")

        # Generate a response using the OpenAI API
        answer = get_openai_answer(question, text_content)
        print(f"Generated answer: {answer}")

        prompts_and_responses.append({
            "prompt": question,
            "response": answer
        })

    print(f"Finished processing {filename}\n")

# Save prompts and responses to a file
with open(output_file, "w", encoding="utf-8") as file:
    for item in prompts_and_responses:
        file.write(f"Prompt: {item['prompt']}\nResponse: {item['response']}\n\n")

print(f"Prompts and responses saved to {output_file}.")
