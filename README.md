# Mobile User Behavior Prediction App

This **Streamlit** web application predicts user behavior based on mobile usage data. It uses a trained neural network model (ANN) to classify user behavior and display the results with an interactive interface.

## 🛠 Features

- **User Input**: Collects features such as device model, operating system, app usage time, screen on time, and more.
- **Data Preprocessing**: Utilizes a preprocessing pipeline that includes one-hot encoding and scaling.
- **Model Prediction**: Uses a pre-trained artificial neural network (ANN) to predict user behavior class (1-5).
- **Progress Bar**: Displays the predicted class as a colored progress bar based on the behavior class.
- **Streamlit Interface**: Easy-to-use interface for inputting mobile usage data and viewing predictions.

## 🚀 Getting Started

### Prerequisites

To run this project locally, you'll need:

- **Python 3.7+**
- **pip** (Python package manager)
- **Streamlit** for the web interface

### Installation

1. Create a virtual environment:
    ```bash
    python -m venv env
    ./env/Scripts/activate
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the Streamlit app, use the following command:

```bash
streamlit run main.py
```

## 🔧 How It Works

1. **User Input**: The app prompts users to input features such as the operating system, device model, app usage time, screen on time, etc.
2. **Data Preprocessing**: The input features are one-hot encoded, scaled, and transformed before being passed to the model for prediction.
3. **Model Prediction**: The app uses a pre-trained neural network model to classify the user behavior.
4. **Results Display**: The predicted behavior class (ranging from 1 to 5) is shown on the screen, along with a colored progress bar for visual feedback.

## 📊 Model Used

- **Neural Network (ANN)**: The model consists of three layers, including two hidden layers with ReLU activation and an output layer with a softmax activation function for multi-class classification.
- **Model Details**:
- **Input Layer**: Takes features such as app usage time, screen on time, battery drain, and more.
- **Hidden Layers**: Two dense layers with 12 units each and ReLU activation.
- **Output Layer**: A dense layer with 5 units (for 5 behavior classes) and softmax activation.


## 📂 Directory Structure
```
Mobile-User-Behavior-Prediction/
├── data/
│   └── user_behavior_dataset.csv
├── model/
│   └── user_behavior_model.keras
│   └── scaler.pkl
├── main.py
├── requirements.txt
└── train.py
```

### File Descriptions:

- `train.py`: Script used for training the model with user behavior data.
- `main.py`: The Streamlit app that allows users to input data and make predictions.
- `requirements.txt`: Contains a list of required Python packages.

## 🔧 Dependencies

All required dependencies are listed in the `requirements.txt` file. You can install them using:
```
pip install -r requirements.txt
```

### Key dependencies:

`streamlit`
`pandas`
`numpy`
`scikit-learn`
`tensorflow`
`matplotlib`

## 📸 Screenshots

### App Interface:


### Prediction Results:

### Confusion Metrics:

