from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
from collections import defaultdict
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get port from environment variable or default to 80
port = int(os.environ.get('PORT', 80))

# Update the model path to match Azure File Share mount point
MODEL_PATH = os.getenv('RECOMMENDATION_MODEL_PATH', '/mnt/model/moviesModel.pkl')

# Global variables for model and trainset
model = None
trainset = None

def check_mount_status():
    """Check the status of the mounted file share"""
    try:
        mount_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(mount_dir):
            logger.info(f"Mount directory exists: {mount_dir}")
            logger.info(f"Contents of mount directory: {os.listdir(mount_dir)}")
        else:
            logger.error(f"Mount directory does not exist: {mount_dir}")
    except Exception as e:
        logger.error(f"Error checking mount status: {str(e)}")

def load_model():
    """Load the model from mounted file share"""
    global model, trainset
    try:
        logger.info(f"Starting model loading process from {MODEL_PATH}")
        check_mount_status()
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
            
        logger.info(f"Model file found, attempting to load...")
        
        with open(MODEL_PATH, 'rb') as f:
            model, trainset = pickle.load(f)
            
        logger.info("Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Trainset type: {type(trainset)}")
        return True
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return False

def get_top_n_recommendations(user_id, n=5):
    """Get top N recommendations for a user"""
    if user_id not in trainset.all_users():
        return []
    
    # Get items the user hasn't rated
    inner_uid = trainset.to_inner_uid(user_id)
    user_items = set([j for (j, _) in trainset.ur[inner_uid]])
    all_items = set(trainset.all_items())
    items_to_predict = list(all_items - user_items)
    
    # Predict ratings for all items
    predictions = []
    for item_id in items_to_predict:
        pred = model.predict(user_id, trainset.to_raw_iid(item_id))
        predictions.append((trainset.to_raw_iid(item_id), pred.est))
    
    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:n]

@app.route('/users', methods=['GET'])
def get_users():
    """Endpoint to get information about users in the dataset"""
    try:
        if model is None or trainset is None:
            # Try to load the model if it's not loaded
            if not load_model():
                return jsonify({'error': 'Model not loaded properly'}), 500

        sample_users = list(trainset.all_users())[:10]
        users_info = []
        for inner_uid in sample_users:
            raw_uid = trainset.to_raw_uid(inner_uid)
            users_info.append({
                'inner_id': int(inner_uid),
                'raw_id': str(raw_uid),
                'ratings_count': len(trainset.ur[inner_uid])
            })
        
        return jsonify({
            'total_users': len(trainset.all_users()),
            'sample_users': users_info
        })
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        mount_dir = os.path.dirname(MODEL_PATH)
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None and trainset is not None,
            'mount_status': {
                'directory_exists': os.path.exists(mount_dir),
                'directory_contents': os.listdir(mount_dir) if os.path.exists(mount_dir) else [],
                'model_file_exists': os.path.exists(MODEL_PATH)
            },
            'model_path': MODEL_PATH
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    """Get recommendations endpoint"""
    try:
        # Get userId from query parameters
        user_id = request.args.get('userId')
        
        if not user_id:
            return jsonify({'error': 'userId parameter is required'}), 400
            
        if model is None or trainset is None:
            # Try to load the model if it's not loaded
            if not load_model():
                return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Get recommendations
        try:
            recommendations = get_top_n_recommendations(int(user_id))
            if not recommendations:
                return jsonify({
                    'error': f'No recommendations found for user {user_id}'
                }), 404
                
            return jsonify({
                'userId': user_id,
                'recommendations': [
                    {
                        'movie_id': int(movie_id),
                        'predicted_rating': float(rating)
                    }
                    for movie_id, rating in recommendations
                ]
            })
        except ValueError as ve:
            return jsonify({'error': f'User {user_id} not found in the training set'}), 404
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Load model at startup
if not load_model():
    logger.warning("Failed to load model at startup - will try loading on first request")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=port) 