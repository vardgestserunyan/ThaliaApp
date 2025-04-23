from flask import Flask, request, jsonify
from movie_summary import MovieSummarizer

thalia_app = Flask(__name__)

@thalia_app.route("/api/movie_summarizer", methods=["POST"])
def movie_summarizer():
    input = request.get_json()
    movie_name = input["movie_name"]
    movie_smrzr = MovieSummarizer()
    summary = movie_smrzr(movie_name)
    output = {"movie_summary": summary}

    return jsonify(output)


if __name__ == "__main__":
    app_params = {"host": "0.0.0.0", "port": "8080", "debug": True}
    thalia_app.run(**app_params)