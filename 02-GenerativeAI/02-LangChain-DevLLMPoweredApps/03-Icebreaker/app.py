
from logging import debug
from flask import Flask, render_template, request, jsonify
from Icebreaker import ice_break_with

# Initialize the Flask runtime
app = Flask(__name__)





@app.route("/")
def index():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=4444)
