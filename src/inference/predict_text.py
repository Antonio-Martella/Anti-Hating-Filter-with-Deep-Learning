import os, sys
from core import HateModelInference

if __name__ == "__main__":
    model = HateModelInference()
    
    while True:
        text = input("Scrivi una frase ('exit' per uscire'): ")
        if text == "exit": break
        
        result = model.predict_text(text)
        print(result)

