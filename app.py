from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation_logic import recommend_assessments

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data["query"]
    results = recommend_assessments(query)
    
    # Format response to match API spec
    formatted_results = []
    for test in results:
        formatted_results.append({
            "url": test["url"],
            "adaptive_support": test["adaptive_support"],
            "description": test.get("description", "No description available"),  # Add to CSV if needed
            "duration": test["duration"],
            "remote_support": test["remote_support"],
            "test_type": test["test_type"]
        })
    return jsonify({"recommended_assessments": formatted_results}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)