from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('ElderlyModel.pkl', 'rb'))

@app.route('/')
def index():
    return 'test'


@app.route('/api/v1/ktm/final_status', methods=['POST'])
def ckd():
    output = {'output': '',
              }
    int_features = [x for x in request.form.values()]
    print(int_features)
    final = np.array(int_features)
    data_unseen = pd.DataFrame(
        [final], columns=['gender', 'age', 'kidney_period', 'blood_pressure', 'nausea','vomit', 'loss_of_appetite', 'itching', 'hiccups','metallic_taste','fatigue', 'sleeping_difficulty', 'urinate_change', 'mental_sharpness', 'muscle_twitches', 'swelling', 'hypertension']
)



    prediction = model.predict(data_unseen)
    print(prediction[0])

    output = {'final_status': str(prediction[0]),
              }
    return output




if __name__ == "__main__":
    app.run(debug=True)
