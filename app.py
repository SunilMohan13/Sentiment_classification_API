from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel
from jupyter_client.consoleapp import app_aliases

app = Flask(__name__)
api = Api(app)

model = NLPModel()

clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)
    
vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)
    
#argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        
        # vectorize the user's query and make the prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
        
        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_test = 'Negative'
        else:
            pred_test = 'Positive'
        
        # round the predict proba and set to new variable
        confidence = round(pred_proba[0], 3)
    
        # create JSON Object
        output = {'prediction': pred_test, 'confidence':confidence}
    
        return output

# Setup the Api resource here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)