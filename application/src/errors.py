from flask import jsonify
from flask_limiter.errors import RateLimitExceeded


def register_error_handlers(app):
    @app.errorhandler(RateLimitExceeded)
    def ratelimit_handler(e):
        return jsonify(
            {
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please wait a minute before trying again.",
            }
        ), 429

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(
            {"error": "Not Found", "message": "The requested endpoint does not exist."}
        ), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(
            {
                "error": "Internal Server Error",
                "message": "Something went wrong on our end while processing your request. Please try again later.",
            }
        ), 500
