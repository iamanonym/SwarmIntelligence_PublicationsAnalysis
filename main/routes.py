from flask import jsonify
from main.app import app
from main.utils import build_country_graph, build_keyword_graph

"""
    Here are represented API-services those are available 
    for building the graph models based on csv-data file.
"""

@app.route('/countries/<int:width>/<int:height>/')
def countries_api(width, height):
    build_country_graph(width, height)
    return app.send_static_file("countries.svg")


@app.route('/keywords/<int:width>/<int:height>/')
def keywords_api(width, height):
    build_keyword_graph(width, height)
    return app.send_static_file("keywords.svg")
