import os, sys
from core import HateModelInference

if __name__ == "__main__":
  model = HateModelInference()
  while True:
    text = input("Write a sentence ('exit' for to go out): ")
    if text == "exit": break
        
    result = model.predict_text(text)
    print(result)