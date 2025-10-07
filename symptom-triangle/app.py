from flask import Flask, request, render_template, jsonify
from symptom_classifier import classify_symptoms

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    symptoms = data.get("symptoms", "")
    departments = classify_symptoms(symptoms)
    return jsonify({"departments": departments})

if __name__ == "__main__":
    app.run(debug=True)