from flask import Flask,render_template,request
import pickle
import numpy as np

with open("Heart_Stroke_Risk.pkl",'rb') as f:
    model = pickle.load(f)
    

#create an object instance
app =Flask(__name__)


@app.route('/')#by default methods = ['GET']
def new():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    age=int(request.form['age'])
    sex=int(request.form['sex'])
    cholesterol = int(request.form['cholesterol'])
    heart_rate=int(request.form['heart_rate'])
    diabetes=int(request.form['diabetes'])
    family_history=int(request.form['family_history'])
    smoking=int(request.form['smoking'])
    obesity=int(request.form['obesity'])
    alcohol_consumption=int(request.form['alcohol_consumption'])
    exercise_hours_per_week = int(request.form['exercise_hours_per_week'])
    diet=int(request.form['diet'])
    previous_heart_problems=int(request.form['previous_heart_problems'])
    medication_use=int(request.form['medication_use'])
    stress_level=int(request.form['stress_level'])
    sedentary_hours_per_day=int(request.form['sedentary_hours_per_day'])
    bmi=int(request.form['bmi'])
    triglycerides=int(request.form['triglycerides'])
    physical_activity_days_per_week=int(request.form['physical_activity_days_per_week'])
    sleep_hours_per_day=int(request.form['sleep_hours_per_day'])
    country=int(request.form['country'])
    continent=int(request.form['continent'])
    hemisphere=int(request.form['hemisphere'])
    systolic=int(request.form['systolic'])
    diastolic=int(request.form['diastolic'])

    input_data = np.array([[age,sex,cholesterol,heart_rate,diabetes,family_history,smoking,obesity,
                       alcohol_consumption,exercise_hours_per_week,diet,previous_heart_problems,
                       medication_use,stress_level,sedentary_hours_per_day,bmi,triglycerides,
                       physical_activity_days_per_week,
                       sleep_hours_per_day,country,continent,hemisphere,systolic,diastolic]])
    prediction_model=model.predict(input_data)[0]
    if prediction_model==1:
        prediction='Yes!,You May Get Heart Attack'
    else:
        prediction='No,You Are Safe'
    return render_template("index.html",prediction=prediction)
    
    
if __name__ == "__main__":
    app.run(debug=True)

