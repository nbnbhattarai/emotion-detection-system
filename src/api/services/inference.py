'''Prediction model'''

from api.settings import shared_components

from models.inference import naive_inference
from flask import jsonify


def classify_text():
    '''
    Classify the text using given model name
    '''

    print(vars(request))

    db = shared_components['db']
    collection = db.inference

    if request.json:
        json_data = request.json
    else:
        raise TypeError('Invalid request format.')

    record_id = collection.insert(json_data)

    text_data = json_data.text

    classification = naive_inference(text_data)

    return jsonify(classification)
