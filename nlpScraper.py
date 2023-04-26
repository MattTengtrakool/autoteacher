import requests
import re
from bs4 import BeautifulSoup
import os

def extract_text(element):
    if element.name in {"script", "style"}:
        return ""
    if isinstance(element, str):
        return element.strip()
    
    text = ""
    
    if element.name == "h1":
        text += "# "
    elif element.name == "h2":
        text += "## "
    elif element.name == "h3":
        text += "### "
    elif element.name == "h4":
        text += "#### "
    elif element.name == "h5":
        text += "##### "
    elif element.name == "h6":
        text += "###### "
    elif element.name == "li":
        text += "- "
    elif element.name == "code":
        text += "code: "
    
    for child in element.children:
        text += extract_text(child)
        if text and text[-1] not in {".", ":", ";", ","}:
            text += " "
    
    if element.name in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "code"}:
        text += "\n"
    
    return text.strip()

url = "https://cs50.harvard.edu/college/2022/fall/notes/1/"

response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.content, "html.parser")
main_element = soup.find("main")

# Extract text content
text_content = extract_text(main_element)

# Split text content into chunks of maximum length 2048 (OpenAI prompt token size)
chunk_size = 2048

# Split text content into sentences
sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text_content)

# Initialize list of chunks
chunks = []

# Initialize current chunk
current_chunk = ""

# Loop over sentences
for sentence in sentences:
    # If adding the sentence to the current chunk would exceed the maximum length
    if len(current_chunk) + len(sentence) > chunk_size:
        # Add the current chunk to the list of chunks
        chunks.append(current_chunk)
        # Start a new chunk with the sentence
        current_chunk = sentence + " "
    else:
        # Add the sentence to the current chunk
        current_chunk += sentence + " "

# Add the final chunk to the list of chunks
chunks.append(current_chunk)

# Create a folder for the output files
if not os.path.exists("output_nlp"):
    os.makedirs("output_nlp")

# Write chunks to separate text files in the output folder
for i, chunk in enumerate(chunks):
    file_path = os.path.join("output_nlp", f"output_nlp_{i+1}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(chunk)
    print(f"Chunk {i+1} saved to {file_path}.")

print("All content saved to separate text files in the output_nlp folder.")
