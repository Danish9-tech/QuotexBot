from flask import Flask

# Create a Flask application
app = Flask(__name__)

# Register the live verification dashboard blueprint
try:
    from dashboard.web import bp as dashboard_bp
    app.register_blueprint(dashboard_bp)
except Exception as _dash_err:
    # Don't fail the existing health endpoint if dashboard init errors
    import logging
    logging.getLogger("app").warning(f"dashboard blueprint not registered: {_dash_err}")

# Define a route for the homepage
@app.route('/')
def hello_world():
    return 'This bot is made by @Cybrion and currently it hosted and live for everyone'

# Run the application
if __name__ == '__main__':
    app.run()
