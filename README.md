# coronavirus-screener
Well all it asks, are your symptoms to diagnose your risk toward covid-19
# clean_code.py
the brain of the application 

## How to run
### windows
- Download the repo install 
- run command "pip install -r requirements.txt"
- run _"clean_code.py"_
### Linux
- Clone the Repo using "git clone https://github.com/dev-Roshan-lab/covid-screener.git"
- run command "pip install -r requirements.txt"
- run _"clean_code.py"_

### API request
- Made an API through which the process of classification is done faster
- the Flask is hosted on heroku [here](https://covid-screener.herokuapp.com/rf/Other-blood-group/Child/Yes/Yes/Yes) 
- Above Url is `https://covid-screener.herokuapp.com/rf/Other-blood-group/Child/Yes/Yes/Yes` 
- # Explanation of the url
  - the `/rf/` denotes that its a RandomForest Classifier and no other Classifiers avbl 
  - In `/Other-blood-group/` directory :
    - _Other-blood-group_
    - _O-blood-group_
  - In `/Child/` directory :
    - _Child_
    - _Adult_
    - _Older-adult_
  - In all three directories `/Yes/Yes/Yes` 
    - _Yes_
    - _No_
  
  these values can be passed on their respective positions 
  
  - The API returns result in **text** format and **not in Json** Format
  
  - This can Effectively Utilized by Using an [Android App](https://github.com/dev-Roshan-lab/symptom-predictor-api/tree/master/Android%20app%20source) similar to this
  

### made with python 3.8
