import pickle
import pandas as pd
import numpy as np
from flask import *

# create flask app instance
application = Flask(__name__, template_folder = 'templates', static_folder = 'static')

# import all models and the scaler
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))
ridge = pickle.load(open('./models/fwi-predictors/ridge.pkl', 'rb'))
lasso = pickle.load(open('./models/fwi-predictors/lasso.pkl', 'rb'))
elasticnet = pickle.load(open('./models/fwi-predictors/elasticnet.pkl', 'rb'))
gridsearch = pickle.load(open('./models/class-predictors/gridsearch.pkl', 'rb'))
randomsearch = pickle.load(open('./models/class-predictors/randomsearch.pkl', 'rb'))

@application.route('/', methods = ['GET'])
def welcome():
    return render_template('homepage.html')

@application.route('/predictform', methods = ['GET'])
def query_parameters():
    return render_template('userinput.html')

@application.route('/predictdata', methods = ['POST'])
def predict_result():
    # extract input parameters from html form
    linear_model = request.form.get('Linear_Model', None)
    col_names = ['Temperature', 'RH', 'WS', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']

    input = pd.DataFrame([[request.form.get(x) for x in col_names]], columns = col_names)
    input = pd.DataFrame(scaler.transform(input), columns = ['Temperature', 'RH', 'WS', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI'])

    # select the linear model and do the fwi prediction
    fwi = 0.0
    if linear_model == 'ridge':
        fwi = ridge.predict(input)[0]
    elif linear_model == 'lasso':
        fwi = lasso.predict(input)[0]
    else:
        fwi = elasticnet.predict(input)[0]
    fwi = f'FWI = {fwi}'

    # select the logistic model and do class prediction
    fire = 0
    logistic_model = request.form.get('Logistic_Model', None)
    if logistic_model == 'grid':
        fire = gridsearch.predict(np.array(input))[0]
    else:
        fire = randomsearch.predict(np.array(input))[0]
    verdict = f'Verdict : {'Fire' if fire else 'No Fire'}'
    
    return render_template('userinput.html', result = (fwi, verdict))

if __name__ == '__main__':
    application.run(host = '0.0.0.0', port = 8080, debug = True)