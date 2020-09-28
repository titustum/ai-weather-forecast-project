from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/towns', methods=['GET', 'POST'])
def town():
	if request.method=="POST":
		town=request.form.get("search_city")
		print(town)
		return render_template('town.html', town=town,
		temp_preds=[23,24,31,25,26,28,25,29,30,32],
		humid_preds=[67,67,61,25,56,28,20,70,60,32],
		press_preds=[1023,1024,1031,1025,1026,1028,1025,1029,1030,1032],
		windspd_preds=[3,4,1,5,6,18,15,9,0,12],
		winddir_preds=[123,214,331,254,264,281,25,249,130,222],
		desc_preds=["Scattered clouds","Rainy","Few clouds","Thunderstorms","Clear Sky","Clear sky","Thunderstorms","Haze","Heavy clouds","Clear sky"],
		icons_preds=["scattered clouds.png","rainfall_icon.png","few clouds.png","thunderstorm.png","clear sky.png","clear sky.png","thunderstorm.png","haze weather.png","Overcast.jfif","clear sky.png"]
			)   

app.run(host='0.0.0.0', port=5000, debug=True)