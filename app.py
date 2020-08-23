from flask import Flask,render_template,url_for,flash,redirect
from flask import request
import pickle
import numpy as np
import os
import tensorflow as tf
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras.layers as KL


app = Flask(__name__,template_folder='template')

app.config['SECRET_KEY'] = '110971d7541b8b5dc86a9398ff2ac719'

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'


model_cancer = pickle.load(open("random_forest_cancer_model.pkl","rb"))
model_diabetes = pickle.load(open("random_forest_diabetes_model.pkl","rb"))
model_heart = pickle.load(open("random_forest_heart_model.pkl","rb"))
model_kidney = pickle.load(open("random_forest_kidney_model.pkl","rb"))
model_liver = pickle.load(open("random_forest_liver_model.pkl","rb"))
malaria_model = load_model("malariavgg19.h5",custom_objects={'BatchNorm':KL.BatchNormalization}, compile=True)
@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/cancer')
def cancer():
    	return render_template("cancer.html")

@app.route('/diabetes')
def diabetes():
    	return render_template("diabetes.html")

@app.route('/heart')
def heart():
    	return render_template("heart.html")

@app.route('/kidney')
def kidney():
    	return render_template("kidney.html")

@app.route('/liver')
def liver():
    	return render_template("liver.html")   

@app.route('/Malaria')
def Malaria():
    	return render_template("malaria.html") 

def api(full_path):
    data = image.load_img(full_path, target_size=(224,224,3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = malaria_model.predict(data)
    return predicted

@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))


"""@app.route('/upload', methods=['GET','POST'])
def uploading():
	if request.method == 'GET':
		return render_template("malaria.html")
	else:
		try:
			file = request.files['image']
			full_name = os.path.join('uploads',file.filename)
			file.save(full_name)
			data = image.image_load(full_name,target_sizes=[224,224,3])
			data = image.img_to_array(data)
			data = np.expand_dims(data, axis=0)
			data = data*1.0/255
			prediction = malaria_model.predict(data)
			prediction = np.argmax(prediction,axis=1)
			if prediction == 0:
				result = "The Person is Infected With Malaria"
			else:
				result = "The Person is not Infected With Pneumonia"
			return render_template("predict.html",image_file_name = file.filename,result=result)	
		except:
			flash("please upload image first!! ","danger")
			return render_template("malaria.html")"""

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename) 


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        result = model_diabetes.predict(to_predict)
    elif(size==30):#Cancer
        result = model_cancer.predict(to_predict)
    elif(size==12):#Kidney
        result = model_kidney.predict(to_predict)
    elif(size==10):#Liver
        result = model_liver.predict(to_predict)
    elif(size==11):#Heart
        result = model_heart.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):#Kidney
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):#Heart
            result = ValuePredictor(to_predict_list,11)
        elif(len(to_predict_list)==10):#Liver
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))

if __name__ == '__main__':
    app.run(debug=True)    

