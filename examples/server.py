"""Flask server for API access"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from src import DeterministicMPSModel, get_m4_config
import logging

app = Flask(__name__)

# Initialize model globally
config = get_m4_config()
model = DeterministicMPSModel(config['model'])

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        
        output, elapsed = model.generate(prompt, max_new_tokens=max_tokens)
        
        return jsonify({
            'choices': [{
                'text': output,
                'index': 0,
                'finish_reason': 'stop'
            }],
            'model': config['model'],
            'usage': {
                'completion_tokens': len(output.split()),
                'prompt_tokens': len(prompt.split()),
                'total_tokens': len(output.split()) + len(prompt.split())
            },
            'processing_time': elapsed
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': config['model'],
        'device': 'mps'
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(f"Starting server with model: {config['model']}")
    app.run(host='0.0.0.0', port=8000, threaded=False)