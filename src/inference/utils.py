import re

def clean(text):
  if not isinstance(text, str):
    return ""
  # Lowercase
  text = text.lower()
  # Remove URLs
  text = re.sub(r'http\S+|www\S+|https\S+', '', text)
  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)
  # Remove mentions (@username)
  text = re.sub(r'@\w+', '', text)
  # Remove weird symbols but keep punctuation like ! ? . ,
  text = re.sub(r"[^a-zA-Z0-9.,!?\'\"\s]", " ", text)
  # Replace multiple spaces
  text = re.sub(r'\s+', ' ', text).strip()
  
  return text