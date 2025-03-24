from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.abspath(os.path.join('model', 'moviesModel.pkl'))
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    try:
        # Get userId from query parameters
        user_id = request.args.get('userId')
        
        if not user_id:
            return jsonify({'error': 'userId parameter is required'}), 400
            
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
            
        # Get recommendations from the model
        recommendations = model.get_recommendations(int(user_id))
        
        return jsonify({
            'userId': user_id,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 