# A3 - Car Price Prediction: Web-Based Application using Machine Learning Model and connecting Github and Server using CICD 

This project is a web app that uses a machine learning model with Dash by Plotly. It is designed to have a visually appealing interface while staying functional. If users skip any form fields, the system automatically fills in the missing values.

## Project Structure

- `A3_PredictingCarPrice`: Consists of all the .ipynb and all related to this project.
- `app.py`: Main application file.
- `app/`: Folder containing pages UI.
- `model/`: Folder containing the machine learning model.
- `test_prediction.py`: CICD test file.
- `test/`: Folder containing test file while doing CICD.
- `.Dockerfile`: Installs list of dependencies needed to run the application.
- `docker-compose.yaml`: Needed to run the dockerize the container.
- `Screenshots`: Consists of images of mlflow environment, mlflow model and Ui Image that is runned in ml server.
- `README.md`: Project documentation.
- `requirements.txt`: Consists of all the name and version of .

## Usage

Once the app is running, users can enter data through the web interface. If they leave some fields empty, the system fills in the missing values automatically and makes predictions using the built-in machine learning model.

## Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/enisha3/ML_A3_PredictionCarPrice_Inisha.git

2. Navigate into the project directory:
   ```bash
   cd <yourproject_dir>

3. Run Dockerfile:
   ```bash
   docker compose up

4. Run the application:
   ```bash
    python3 app/app.py

## RUNNING APPLICATION:
mlflow logs: https://mlflow.ml.brain.cs.ait.ac.th/<experiment_name: st125563-a3>

docker hub image: https://hub.docker.com/repository/docker/inisha5563/car_price_prediction/general

brain lab server: https://inishacarpricepredictor-st125563.ml.brain.cs.ait.ac.th/

## Application running on server images:
![Imageone](screenshots/mlflow_environment.png)

![Imagetwo](screenshots/mlflow_model.png)

![Imagethree](screenshots/UI_Design.png)