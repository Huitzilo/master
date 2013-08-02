from flask import Flask, render_template, jsonify
import pickle
import os

DEBUG = True
app = Flask(__name__)
app.debug = DEBUG

data_path = os.path.join(os.path.dirname(__file__), 'models')
data = pickle.load(open(os.path.join(data_path, 'predictions.pkl')))
methods = data.keys()
receptors = sorted(data.values()[0].keys(), cmp=lambda x, y: cmp(x.lower(), y.lower()))


@app.route('/')
def index():
    return render_template('layout.html', receptors=receptors, methods=methods)


@app.route('/<receptor>/<method>')
def receptor_overview(receptor, method):
    """return all predictions for one receptor method combination"""
#    print receptor, method
    predictions = data[method][receptor]['predictions']
    sorted_predictions = sorted(predictions, key=lambda x: x['prediction'], reverse=True)
    return jsonify(predictions=sorted_predictions,
                   q2_score=data[method][receptor]['score'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
