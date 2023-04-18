import os
from google.cloud import language_v1

# Set the environment variable for Google API authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/matt/downloads/vaulted-bus-383605-bcb39e1c743c.json"

def analyze_entities(text_content):
    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    document = language_v1.Document(content=text_content, type_=type_, language="en")

    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entities(request={'document': document, 'encoding_type': encoding_type})

    return response

# Read the text_content from the output_nlp.txt file
with open("output_nlp.txt", "r", encoding="utf-8") as file:
    text_content = file.read()

# Analyze the text_content
response = analyze_entities(text_content)

prompts_and_responses = []

for entity in response.entities:
    # Use the entity name as a prompt
    prompt = entity.name

    # Find the response by taking the entity's mention text
    for mention in entity.mentions:
        response = mention.text.content

        prompts_and_responses.append({
            "prompt": prompt,
            "response": response
        })

# Save prompts and responses to a file
with open("prompts_and_responses.txt", "w", encoding="utf-8") as file:
    for item in prompts_and_responses:
        file.write(f"Prompt: {item['prompt']}\nResponse: {item['response']}\n\n")

print("Prompts and responses saved to prompts_and_responses.txt.")
