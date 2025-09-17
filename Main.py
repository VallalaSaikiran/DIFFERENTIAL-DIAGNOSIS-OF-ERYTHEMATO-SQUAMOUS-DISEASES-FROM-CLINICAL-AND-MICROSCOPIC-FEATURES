from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

from IPython.display import display
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
import os, pickle, joblib
from PIL import Image, ImageTk

accuracy = []
precision = []
recall = []
fscore = []

categories=['psoriasis','seboreic dermatitis','lichen planus','pityriasis rosea','cronic dermatitis','pityriasis rubra pilaris' ]
target_name  ='class'
model_folder = "model"

def Upload_Dataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")
    label = dataset.groupby('class').size()
    label.plot(kind="bar")
    plt.title("Various Class Type Graph")
    plt.show()

def Preprocess_Dataset():
    global dataset, X_resampled, y_after, feature_names
    text.delete('1.0', END)
    
    dataset.replace('?', np.nan, inplace=True)
    dataset.dropna(inplace =True)

    df = dataset.copy()
    
    le= LabelEncoder()
    dataset['class']=le.fit_transform(dataset['class'])
    dataset
    
    X = dataset.iloc[:, 0:34]
    y = dataset.iloc[:, -1]
    
    text.insert(END, "Data preprocessed successfully.\n\n")
    text.insert(END, "Dataset before label encoding:\n" + str(df.head()) + "\n\n")
    text.insert(END, "Dataset after label encoding:\n" + str(X.head()) + "\n\n")
    text.insert(END, "Dataset description:\n" + str(dataset.describe()) + "\n\n")
    
    feature_names = X.columns.tolist()

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_after = smote.fit_resample(X, y)
    
    labels = [
        'psoriasis',
        'seboreic dermatitis',
        'lichen planus',
        'pityriasis rosea',
        'chronic dermatitis',
        'pityriasis rubra pilaris'
    ]
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b'   # chestnut brown
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before SMOTE plot
    sns.countplot(data=df, x='class', palette=colors, ax=axes[0])
    axes[0].set_title('Erythemato Squamous Classes Before SMOTE')
    axes[0].set_xlabel('Erythemato Squamous Class')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    for p in axes[0].patches:
        height = int(p.get_height())
        axes[0].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height + 1),
                        ha='center', va='center', fontsize=10, fontweight='bold')

    # After SMOTE plot
    # Map numeric values to class names if y_after is numeric
    y_after_named = [labels[i] for i in y_after]

    sns.countplot(x=y_after_named, palette=colors, ax=axes[1])
    axes[1].set_title('Erythemato Squamous Classes After SMOTE')
    axes[1].set_xlabel('Erythemato Squamous Class')
    axes[1].set_ylabel('Count')
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    for p in axes[1].patches:
        height = int(p.get_height())
        axes[1].annotate(f'{height}', (p.get_x() + p.get_width() / 2., height + 1),
                        ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def Train_Test_Splitting():
    global X, Y, dataset, feature_names, X_resampled, y_after 
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X_resampled.shape[0]) + "\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_after, test_size=0.2)
    text.insert(END, "Dataset Train and Test Split" + "\n")
    text.insert(END, "Total records found in dataset to train: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records found in dataset to test: " + str(X_test.shape[0]) + "\n")    

def calculateMetrics(algorithm, predict, y_test):
    labels = categories
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")
    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

    
def existing_classifier1():
    global X_train, X_test, y_train, y_test, feature_names
    
    text.delete('1.0', END)
    model = 'model'
    model_name = 'DTC.pkl'
    path = os.path.join(model, model_name)
    if os.path.exists(path):
        dtc = joblib.load(path)
        print("Model loaded successfully.")
        predict = dtc.predict(X_test)
        calculateMetrics("DTC Model", predict, y_test)
    else:
        dtc = DecisionTreeClassifier(max_depth=3)
        dtc.fit(X_train, y_train)
        joblib.dump(dtc, path) 
        print("Model saved successfuly.")
        predict = dtc.predict(X_test)
        calculateMetrics("DTC Model", predict, y_test)
    root_feature_index = dtc.tree_.feature[0]
    
    root_feature_name = feature_names[root_feature_index]

    text.insert(END, f"Input Features of DTC Model: {feature_names}\n")
    
    text.insert(END, f"Root node splits on feature: {root_feature_name}\n")

    plt.figure(figsize = (15,7))
    plot_tree(dtc,filled = True)
    plt.title(f"Internal Architecture of DTC Model on Erythemato Squamous")
    plt.show()

def existing_classifier2():
    global X_train, X_test, y_train, y_test
    
    text.delete('1.0', END)
    model = 'model'
    model_name = 'SVM.pkl'
    path = os.path.join(model, model_name)
    if os.path.exists(path):
        svc = joblib.load(path)
        print("Model loaded successfully.")
        predict = svc.predict(X_test)
        calculateMetrics("SVM Classifier", predict, y_test)
    else:
        svc = SVC()
        svc.fit(X_train, y_train)
        joblib.dump(svc, path) 
        print("Model saved successfuly.")
        predict = svc.predict(X_test)
        calculateMetrics("SVM Classifier", predict, y_test)

def proposed_classifier3():
    global X_train, X_test, y_train, y_test, mlp
    text.delete('1.0', END)
    
    model = 'model'
    model_name = 'MLP.pkl'
    path = os.path.join(model, model_name)
    if os.path.exists(path):
        mlp = joblib.load(path)
        print("Model loaded successfully.")
        predict = mlp.predict(X_test)
        calculateMetrics("Proposed DL Classifier", predict, y_test)
    else:
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        joblib.dump(mlp, path) 
        print("Model saved successfuly.")
        predict = mlp.predict(X_test)
        calculateMetrics("Proposed DL Classifier", predict, y_test)

def graph():
    #comparison graph between all algorithms
    df = pd.DataFrame([['DTC Model','Accuracy',accuracy[0]],['DTC Model','Precision',precision[0]],['DTC Model','Recall',recall[0]],['DTC Model','FSCORE',fscore[0]],
                       ['SVM Classifier','Accuracy',accuracy[1]],['SVM Classifier','Precision',precision[1]],['SVM Classifier','Recall',recall[1]],['SVM Classifier','FSCORE',fscore[1]],
                       ['DL Classifier','Accuracy',accuracy[2]],['DL Classifier','Precision',precision[2]],['DL Classifier','Recall',recall[2]],['DL Classifier','FSCORE',fscore[2]],                     
                     ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 4))
    plt.title("Performance Evaluation")
    plt.xticks(rotation=360)
    plt.show()        
    
  
def Prediction():
    global mlp, categories

    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')
    test = pd.read_csv(filename)
    test_data_display = test.copy()
    predict = mlp.predict(test)
    for i, prediction in enumerate(predict):
        sample_data = test_data_display.iloc[i]
        formatted_data = ', '.join(f"{col}: {sample_data[col]}" for col in test_data_display.columns)
        text.insert(END, f"Features: {formatted_data}\n")
        pred_label = categories[prediction]  # Corrected this line
        text.insert(END, f"Test Data {i+1}: {pred_label}\n\n")


import tkinter as tk
from tkinter import messagebox
import redis
import hashlib

# Connect to Redis
def connect_redis():
    return redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Hash password before storing in Redis for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before storing
                hashed_password = hash_password(password)

                # Store the user in Redis with multiple field-value pairs
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    messagebox.showerror("Error", "User already exists!")
                else:
                    # Using multiple field-value pairs in hset
                    conn.hset(user_key, "username", username)
                    conn.hset(user_key, "password", hashed_password)
                    conn.hset(user_key, "role", role)
                    messagebox.showinfo("Success", f"{role} Signup Successful!")
                    signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Create the signup window
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    # Username field
    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)
    
    # Password field
    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    # Signup button
    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before checking
                hashed_password = hash_password(password)

                # Check if the user exists in Redis
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    stored_password = conn.hget(user_key, "password")
                    stored_role = conn.hget(user_key, "role")

                    if stored_password == hashed_password and stored_role == role:
                        messagebox.showinfo("Success", f"{role} Login Successful!")
                        login_window.destroy()
                        if role == "Admin":
                            show_admin_buttons()
                        elif role == "User":
                            show_user_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid Credentials!")
                else:
                    messagebox.showerror("Error", "User not found!")
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Disease Dataset", command=Upload_Dataset, font=font1).place(x=100, y=160)
    tk.Button(main, text="Data Preprocessing", command=Preprocess_Dataset, font=font1).place(x=360, y=160)
    tk.Button(main, text="Data Splitting", command=Train_Test_Splitting, font=font1).place(x=580, y=160)
    tk.Button(main, text="Build & Train DTC Model", command=existing_classifier1, font=font1).place(x=800, y=160)
    tk.Button(main, text="Build & Train SVM Classifier", command=existing_classifier2, font=font1).place(x=100, y=210)
    tk.Button(main, text="Build & Train Deep Learning Classifier", command=proposed_classifier3, font=font1).place(x=450, y=210)
   
    tk.Button(main, text="Performance Graph", command=graph, font=font1).place(x=850, y=210)

def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction on Test Data", command=Prediction, font=font1).place(x=650, y=200)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

main = tk.Tk()
#main.geometry("1300x1200")
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.jpg")  
bg_image = bg_image.resize((screen_width, screen_height), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(main, width=screen_width, height=screen_height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title
font = ('times', 18, 'bold')
title_text = "Machine Learning-Based Differential Diagnosis of Erythemato Squamous Diseases from Clinical and Microscopic Features"
title = tk.Label(main, text=title_text, bg='white', fg='black', font=font, wraplength=screen_width - 200, justify='center')
canvas.create_window(screen_width // 2, 50, window=title)

font1 = ('times', 14, 'bold')

# Create text widget and scrollbar
text_frame = tk.Frame(main, bg='white')
text = tk.Text(text_frame, height=22, width=130, font=font1, wrap='word')
scroll = tk.Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.grid(row=0, column=0, sticky='nsew')
scroll.grid(row=0, column=1, sticky='ns')
text_frame.grid_rowconfigure(0, weight=1)
text_frame.grid_columnconfigure(0, weight=1)

# Position the text_frame on the canvas, centered horizontally
canvas.create_window(screen_width // 2, 300, window=text_frame, anchor='n')


# Admin and User Buttons
font1 = ('times', 14, 'bold')

tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=25, height=1, bg='thistle').place(x=50, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=25, height=1, bg='thistle').place(x=400, y=100)

admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=25, height=1, bg='lightsteelblue')
admin_button.place(x=750, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=25, height=1, bg='lightsteelblue')
user_button.place(x=1100, y=100)

main.config(bg='lavender')
main.mainloop()
