"""This module will serve the api request."""

from flask import Flask, Response, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from services.user import create_user, fetch_users, fetch_user, update_user, remove_user
from setting import shared_components


def init_app():

    app = Flask(__name__)
    app = create_route(app)

    # Load Config File for DB
    app.config.from_pyfile('config/config.cfg')
    CORS(app)
    mongo = PyMongo(app)

    # Select the database
    db = mongo.db
    shared_components['db'] = db
    return app


def create_route(app):
    """
    Adds different rules to the urls
    """
    app.add_url_rule(rule='/',
                    view_func=get_initial_response, methods=['GET'])
    app.add_url_rule(rule='/api/v1/users',
                     view_func=create_user, methods=['POST'])
    app.add_url_rule(rule='/api/v1/users',
                     view_func=fetch_users, methods=['GET'])
    app.add_url_rule(rule='/api/v1/users/<user_id>',
                     view_func=fetch_user, methods=['GET'])
    app.add_url_rule("/api/v1/users/<user_id>",
                     view_func=update_user, methods=['PUT'])
    app.add_url_rule("/api/v1/users/<user_id>",
                     view_func=remove_user, methods=['DELETE'])
    return app


def get_initial_response():
    """Welcome message for the API."""
    # Message to the user
    message = {
        'api_version': 'v1.0',
        'status': '200',
        'message': 'Welcome to the Flask API'
    }
    # Making the message looks good
    resp = jsonify(message)

    # Returning the object
    return resp


app = init_app()


@app.errorhandler(404)
def page_not_found(e):
    """Send message to the user with notFound 404 status."""
    # Message to the user
    message = "This route is currently not supported."
    " Please refer API documentation."

    # Sending OK response
    # Returning the object
    return Response(message, status=404)