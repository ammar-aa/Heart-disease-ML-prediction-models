# Heart Disease Prediction App

This project is a Streamlit web application that predicts heart disease using patient data.
It also demonstrates deployment using ngrok to expose the local Streamlit app online.

## Features

- Predict heart disease using selected patient features
- Visualize patient data and model results
- Easy deployment using ngrok

## Installation

1. Clone this repository:
   ```bash
   git clone <your_repo_url>
   ```

2. Navigate to the project folder:
   ```bash
   cd <project_folder>
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App Locally

1. Start Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Open your browser at the local URL (default: http://localhost:8501)


## Deployment with ngrok
1. Download ngrok from [https://ngrok.com/download](https://ngrok.com/download)
2. Expose your Streamlit app using:
   ```bash
   ngrok http 8501
   ```
3. Use the public ngrok URL to access your app online


### Optional: ngrok authtoken
- Sign up on ngrok website and get your auth token
- Run:
  ```bash
  ngrok config add-authtoken <your_token>
  ```
- This allows longer sessions and custom subdomains

## Usage
- Input patient data into the form
- Click "Predict" to see the prediction result
- Visualize charts and analysis in the dashboard


## Notes
- Always download ngrok from the official website to avoid malware
- Close ngrok when done to stop exposing your local machine


## License
This project is for educational purposes.


## run this command to install essential
pip install -r requirements.txt
