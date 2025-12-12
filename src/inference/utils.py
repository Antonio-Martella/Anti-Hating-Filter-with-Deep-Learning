import re
import os

def clean(text):

  '''
  Cleans and normalizes a text string for NLP tasks. The function lowercases the input,
  removes URLs, HTML tags, user mentions, unusual symbols, and compresses multiple spaces.

  Parameters
  ----------
  text : str
      Raw input text to be cleaned.

  Returns
  -------
  str
      The cleaned and normalized text. Returns an empty string if the input is not a valid string.
  '''

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


def resolve_path(path_str):
  
  '''
  Resolves a file path by checking whether the given path exists as-is or inside the `data/`
  directory. Returns the first valid path found; otherwise raises FileNotFoundError.

  Parameters
  ----------
  path_str : str
      File path provided by the user.

  Returns
  -------
  str
      The resolved valid file path.

  Raises
  ------
  FileNotFoundError
      If the file is not found in either the given path or `data/`.
  '''

  if os.path.isfile(path_str):
    return path_str
    
  candidate = os.path.join("data", path_str)
  if os.path.isfile(candidate):
    return candidate
    
  raise FileNotFoundError(f"File not found: {path_str} or {candidate}")
