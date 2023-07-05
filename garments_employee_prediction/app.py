import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
import logging
import pandas as pd
import scatter
from sklearn.model_selection import train_test_split


app = Flask(__name__)

data = pd.read_csv('garments_worker_productivity (1).csv')

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

model = pickle.load(open('model_pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # retrieving values from form
    init_features = [str(x) for x in request.form.values()]

    list1 = init_features
    print(init_features)
    import datetime as dt
    '''list1 = ['1/1/2016', 'Quarter2', 'finishing', 'Thursday', '4', '0.2', '12.3',
             '990', '9012', '120', '1', '0', '1', '30']'''

    # 3data1='"01/01/2016"'
    data1_1 = dt.datetime.strptime(list1[0], '%d/%m/%Y').date()

    month = data1_1.month

    day = data1_1.day
    year = data1_1.year

    list2 = []

    list2.append(int(list1[4]))
    list2.append(float(list1[5]))
    list2.append(float(list1[6]))
    list2.append(int(list1[7]))
    list2.append(int(list1[8]))
    list2.append(int(list1[9]))
    list2.append(int(list1[10]))
    list2.append(int(list1[11]))
    list2.append(int(list1[12]))
    list2.append(int(list1[13]))
    list2.append(month)
    list2.append(day)
    list2.append(year)

    with open("onehotencoder_pkl", 'rb') as file:
        myvar = pickle.load(file)
    df = pd.DataFrame([list1[1:4]], columns=['quarter', 'department', 'week'])
    values = myvar.transform(df).toarray()
    for i in range(0, values.shape[1]):
        list2.append(int(values[0][i]))

    list2
    init_features = list2

    final_features = [np.array(init_features)]

    prediction = model.predict(final_features)  # making prediction

    return render_template('index.html',
                           prediction_text='Predicted Value: {}'.format(prediction))  # rendering the predicted result

'''if __name__ == "__main__":
    app.run(debug=True)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # retrieving values from form
    init_features = [str(x) for x in request.form.values()]

    data = init_features

    data['department'] = data['department'].replace(['sweing'], ['sewing'])

    data['wip'].fillna(data['wip'].mean(), inplace=True)

    data.date = pd.to_datetime(data.date, dayfirst=False)

    data.date = data['date'].dt.strftime('%Y-%m-%d')

    data.set_index(data['date'])

    x = data

    x_train, x_test = train_test_split(x, test_size=0.3)

    train = pd.get_dummies(x_train)

    test = pd.get_dummies(x_test)

    final_train, final_test = train.align(test, join='inner', axis=1)

    final_train1 = final_train.drop(columns=['actual_productivity'])

    final_test1 = final_test.drop(columns=['actual_productivity'])

    y_train1 = final_train[['actual_productivity']]

    y_test1 = final_test[['actual_productivity']]

    x_train1 = final_train1

    x_test1 = final_test1

    list1 = init_features
    print(init_features)
    import datetime as dt
    list1 = ['1/1/2016', 'Quarter2', 'finishing', 'Thursday', '4', '0.2', '12.3', '990', '9012', '120', '1', '0', '1',
             '30']

    # 3data1='"01/01/2016"'
    data1_1 = dt.datetime.strptime(list1[0], '%d/%m/%Y').date()

    month = data1_1.month

    day = data1_1.day
    year = data1_1.year

    if list1[1] == "Quarter1":
        quarter = [1, 0, 0, 0, 0]
    elif list1[1] == "Quarter2":
        quarter = [0, 1, 0, 0, 0]
    elif list1[1] == "Quarter3":
        quarter = [0, 0, 1, 0, 0]
    elif list1[1] == "Quarter4":
        quarter = [0, 0, 0, 1, 0]
    else:
        quarter = [0, 0, 0, 0, 1]

    if list1[2] == "others":
        department = [1, 0, 0]
    elif list1[2] == "finishing":
        department = [0, 1, 0]
    else:
        department = [0, 0, 1]

    if list1[3] == "Monday":
        week = [1, 0, 0, 0, 0, 0]
    elif list1[3] == "Saturday":
        week = [0, 1, 0, 0, 0, 0]
    elif list1[3] == "Sunday":
        week = [0, 0, 1, 0, 0, 0]
    elif list1[3] == "Thursday":
        week = [0, 0, 0, 1, 0, 0]
    elif list1[3] == "Tuesday":
        week = [0, 0, 0, 0, 1, 0]
    else:
        week = [0, 0, 0, 0, 0, 1]

    list2 = []

    list2.append(int(list1[4]))
    list2.append(float(list1[5]))
    list2.append(float(list1[6]))
    list2.append(int(list1[7]))
    list2.append(int(list1[8]))
    list2.append(int(list1[9]))
    list2.append(int(list1[10]))
    list2.append(int(list1[11]))
    list2.append(int(list1[12]))
    list2.append(int(list1[13]))
    list2.append(month)
    list2.append(day)
    list2.append(year)
    list2 = list2 + quarter
    list2 = list2 + department
    list2 = list2 + week

    list2
    init_features = list2
    init_features1 = []
    init_features2 = pd.to_datetime(init_features[0].date, dayfirst=False)
    month =pd.DataFrame(init_features2.date.dt.month)
    day =pd.DataFrame(init_features2.date.dt.day)
    print(month)
    print(day)

    final_features = [np.array(init_features)]

    prediction = model.predict(final_features)

    return render_template('index.html',
                           prediction_text='Predicted Actual Productivity: {}'.format(prediction))
                           
'''

@app.route('/output', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        input_list = request.form.getlist('v')

        graph_selection = request.form.getlist('a')

        if graph_selection[0] == 'scatterplot':

            return (scatter.scatter_plot(input_list[0], input_list[1]))

        elif graph_selection[0] == 'linechart':

            return (scatter.line_plot(input_list[0], input_list[1]))

        else:

            return ('Not printing the box plot')

    return render_template('graph.html')



if __name__ == "__main__":
    app.run(debug=True)
