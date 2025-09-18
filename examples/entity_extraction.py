"""Entity extraction example"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import DeterministicMPSModel
import json

def extract_entities(model, text):
    """Extract entities from text"""
    
    prompt = f"""Extract all named entities from the following text and return them as JSON:

Text: {text}

Format:
{{
    "persons": [],
    "organizations": [],
    "locations": [],
    "products": []
}}

JSON:"""
    
    output, _ = model.generate(prompt, max_new_tokens=200)
    
    try:
        # Parse JSON from output
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            entities = json.loads(output[json_start:json_end])
            return entities
    except:
        pass
    
    return None

def main():
    model = DeterministicMPSModel("Qwen/Qwen2.5-1.5B-Instruct")
    
    test_texts = [
        "Tim Cook, CEO of Apple Inc., announced the new iPhone 15 Pro at the event in Cupertino, California.",
        "Microsoft and OpenAI partnered to bring GPT-4 to Azure cloud services.",
        "The European Union passed new AI regulations in Brussels last week."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        entities = extract_entities(model, text)
        
        if entities:
            print("Extracted entities:")
            print(json.dumps(entities, indent=2))
        else:
            print("Failed to extract entities")

if __name__ == "__main__":
    main()