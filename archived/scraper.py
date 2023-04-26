import requests
from bs4 import BeautifulSoup

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

# Write content to a text file
with open("output_nlp.txt", "w", encoding="utf-8") as file:
    file.write(text_content)

print("Content saved to output_nlp.txt.")
