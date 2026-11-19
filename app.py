"""Run the Flask backend: python app.py"""

from lawbot.api.app import create_app
from lawbot.config import SETTINGS

app = create_app()

if __name__ == "__main__":
    app.run(
        host=SETTINGS.flask_host,
        port=SETTINGS.flask_port,
        debug=SETTINGS.flask_debug,
    )
