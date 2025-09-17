🧑‍⚕️ Machine Learning-Based Differential Diagnosis of Erythemato Squamous Diseases
📌 Overview

This project implements a GUI-based system using Tkinter for the differential diagnosis of erythemato squamous diseases from clinical and microscopic features.
It integrates multiple machine learning classifiers (Decision Tree, SVM, and MLP) with preprocessing, visualization, and prediction functionalities.

The system also supports Admin/User roles with Redis-based login & signup authentication, ensuring secure access.

✨ Features

📂 Upload Dataset and visualize class distributions.

🔄 Data Preprocessing with missing value handling, label encoding, and SMOTE balancing.

✂️ Train-Test Splitting for supervised learning.

🤖 Model Training & Evaluation:

Decision Tree Classifier (DTC)

Support Vector Machine (SVM)

Multi-Layer Perceptron (MLP - Deep Learning)

📊 Performance Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report.

📈 Comparison Graphs across models.

🔐 Authentication: Admin & User role-based login/signup with Redis.

🖥️ User-Friendly GUI with Tkinter.

🔮 Prediction on new test data with detailed feature output.

📁 Project Structure
├── Main.py              # Main application script with GUI, ML pipeline, and authentication
├── model/               # Folder to save trained ML models (DTC.pkl, SVM.pkl, MLP.pkl)
├── Dataset/             # Place your datasets here (CSV format)
├── background.jpg       # GUI background image
└── README.md            # Project documentation

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/erythemato-diagnosis-ml.git
cd erythemato-diagnosis-ml

2️⃣ Install Dependencies

Make sure you have Python 3.7+ installed. Then install requirements:

pip install -r requirements.txt


If you don’t have a requirements.txt yet, here are the required packages:

pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn pillow redis joblib

3️⃣ Start Redis Server

Make sure Redis is running locally (default port 6379).

On Linux/macOS:

redis-server


On Windows, download Redis from Memurai
 or Redis for Windows
.

▶️ Usage

Run the application with:

python Main.py

🔑 User Roles

Admin: Can upload dataset, preprocess, train models, and view performance results.

User: Can log in and run predictions on new test datasets.

📊 Example Workflow

Admin Signup/Login

Upload dataset (CSV)

Preprocess & balance dataset with SMOTE

Split into train & test sets

Train models (DTC, SVM, MLP)

View performance metrics & graphs

Save trained models for reuse

User Login → Load test data → Get predictions

🧪 Models & Evaluation

Decision Tree Classifier (DTC) – interpretable baseline model.

Support Vector Machine (SVM) – effective for high-dimensional data.

Multi-Layer Perceptron (MLP) – deep learning model for complex patterns.

Each model is evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Classification Report
