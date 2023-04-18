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



# Read the text_content from the output_nlp.txt file
with open("output_nlp.txt", "r", encoding="utf-8") as file:
    text_content = file.read()

# Analyze the text_content
response = analyze_entities(text_content)

prompts_and_responses = []

for entity in response.entities:
    # Use the entity name as a prompt
    prompt = f"Please provide more information about {entity.name}"

    # Generate a response using the OpenAI API
    answer = get_openai_answer(prompt, text_content)

    prompts_and_responses.append({
        "prompt": prompt,
        "response": answer
    })

# Save prompts and responses to a file
with open("prompts_and_responses.txt", "w", encoding="utf-8") as file:
    for item in prompts_and_responses:
        file.write(f"Prompt: {item['prompt']}\nResponse: {item['response']}\n\n")

print("Prompts and responses saved to prompts_and_responses.txt.")
