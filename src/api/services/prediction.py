'''Prediction model'''

import logging

from api.settings import shared_components

from models.inference import naive_inference
from flask import jsonify, request

LOGGER = logging.getLogger(__name__)


def classify_text():
    '''
    Classify the text using given model name
    '''
    db = shared_components['db']
    collection = db.inference_requests

    LOGGER.info(f'type of request json: {request.json}')

    if request.json:
        json_data = request.json
    else:
        raise TypeError('Invalid request format.')

    record_id = collection.insert(json_data)

    text_data = json_data['text']

    in_text, pred = naive_inference(text_data)

    collection = db.inferences

    prediction = {
        'request_id': record_id,
        'text': in_text,
        'prediction': pred,
    }

    collection.insert(prediction)

    return jsonify({
        'text': in_text,
        'pred': pred,
    })
