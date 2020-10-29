from flask import Flask
from flask import request
from gpt3_generation import generate_from_prompt

app = Flask(__name__)


@app.route("/gpt3/get", methods=['POST'])
def get_odqa():
    data = request.get_json(force=True)
    print(data)
    return generate_from_prompt(data)


if __name__ == '__main__':
    app.run(port=8989)
