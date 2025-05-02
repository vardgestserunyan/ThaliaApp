from flask import Flask, request, jsonify
from movie_summary import MovieSummarizer
from review_summary import ReviewFetcher, ReviewSummarizer

thalia_app = Flask(__name__)

@thalia_app.route("/api/movie_summarizer", methods=["POST"])
def movie_summarizer():
    input = request.get_json()
    movie_name = input["movie_name"]
    movie_smrzr = MovieSummarizer()
    summary = movie_smrzr(movie_name)
    output = {"movie_summary": summary}

    return jsonify(output)


@thalia_app.route('/api/movie_reviewer', methods=["POST"])
def movie_reviewer():
    input = request.get_json()
    movie_name, order =  input["movie_name"], input["order"]
    review_ftchr, review_smrzr = ReviewFetcher(movie_name), ReviewSummarizer()
    movie_title, (positive_list, negative_list) = review_ftchr.find_and_review(order)
    positive_summary, negative_summary = review_smrzr("positive", positive_list), review_smrzr("negative", negative_list)
    output = {"search_term": movie_name, "found_title": movie_title, 
              "positive": positive_summary, "negative": negative_summary}
    return jsonify(output)

if __name__ == "__main__":
    app_params = {"host": "0.0.0.0", "port": "8080", "debug": True}
    thalia_app.run(**app_params)