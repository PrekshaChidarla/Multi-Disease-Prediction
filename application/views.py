from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tempfile
#create your own views:

def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')

@login_required
def base(request):
    return render(request, 'base.html')

@login_required
def prediction(request):
    return render(request, 'prediction.html')

def about(request):
    return render(request, 'about.html')


# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('home')


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from skimage.transform import resize
from sklearn.model_selection import train_test_split

# Set paths
path = r"Dataset"
model_folder = "model"
os.makedirs(model_folder, exist_ok=True)

# Detect all sub-subfolders as categories
categories = []
for main_dir in os.listdir(path):
    main_path = os.path.join(path, main_dir)
    if os.path.isdir(main_path):
        for sub_dir in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_dir)
            if os.path.isdir(sub_path):
                categories.append(f"{main_dir}/{sub_dir}")
print("✅ Detected categories:", categories)

# File paths
X_file = os.path.join(model_folder, "X.txt.npy")
Y_file = os.path.join(model_folder, "Y.txt.npy")

# Load dataset or preprocess
if os.path.exists(X_file) and os.path.exists(Y_file):
    X = np.load(X_file)
    Y = np.load(Y_file)
    print("✅ X and Y arrays loaded successfully.")
else:
    X = []
    Y = []
    print("\n🚀 Starting image loading process...\n")

    for main_dir in os.listdir(path):
        main_path = os.path.join(path, main_dir)
        if not os.path.isdir(main_path):
            continue

        for sub_dir in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue

            category_label = f"{main_dir}/{sub_dir}"
            print(f"📂 Loading category: {category_label}")

            for img_name in os.listdir(sub_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(sub_path, img_name)
                    try:
                        img_array = cv2.imread(img_path)
                        if img_array is None:
                            continue
                        img_resized = resize(img_array, (64, 64, 3))
                        X.append(img_resized.flatten())  # flattened for traditional ML models
                        Y.append(categories.index(category_label))
                    except Exception as e:
                        print(f"⚠️ Error reading {img_path}: {e}")

    X = np.array(X, dtype='float32') / 255.0  # normalize
    Y = np.array(Y)
    np.save(X_file, X)
    np.save(Y_file, Y)
    print("\n✅ Dataset processed and saved successfully.")

print(f"\n📊 Dataset Summary:")
print(f"Total samples: {len(X)}")
print(f"Unique categories: {len(categories)}")

# Plot class distribution
sns.countplot(x=Y)
plt.title("Class Distribution")
plt.xlabel("Class Index")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Split into train and test sets (no one-hot encoding)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=77)
print("\n📁 Data split into training and testing sets.")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)


def load_data(request):
    return render(request,'prediction.html',{'upload':categories})

from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def DTC_model(request):
    Model_file = os.path.join(model_folder, "DT_Model.pkl")

    if os.path.exists(Model_file):
        dt_classifier = joblib.load(Model_file)
        predict = dt_classifier.predict(x_test)
        print("✅ Decision Tree Classifier model loaded from file.")
        calculateMetrics("Decision Tree Classifier", predict, y_test)
    else:
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(x_train, y_train)
        joblib.dump(dt_classifier, Model_file)
        print("✅ Decision Tree Classifier model trained and model weights saved.")
        predict = dt_classifier.predict(x_test)
        calculateMetrics("Decision Tree Classifier", predict, y_test)

    # Get the latest metrics from the global lists
    latest_accuracy = accuracy[-1] if accuracy else 0
    latest_precision = precision[-1] if precision else 0
    latest_recall = recall[-1] if recall else 0
    latest_fscore = fscore[-1] if fscore else 0

    return render(request, 'prediction.html', {
        'algorithm': 'Decision Tree Classifier',
        'accuracy': round(latest_accuracy, 2),
        'precision': round(latest_precision, 2),
        'recall': round(latest_recall, 2),
        'f1_score': round(latest_fscore, 2),
    })



from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import pickle

def CNN_model(request):
    global num_classes
    num_classes = len(categories)  # Use your categories list here
    
    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel.weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    
    if os.path.exists(Model_file) and os.path.exists(Model_weights):
        # Load the model architecture and weights
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights(Model_weights)
        print("✅ CNN Model loaded from files.")
        
        # Load training history
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
        
        # Predict on test set
        predict_prob = model.predict(X_test)
        predict = predict_prob.argmax(axis=1)
        calculateMetrics("CNN Model", predict, y_test)
        
        # Use accuracy from history (last epoch)
        acc = history['accuracy'][-1] * 100
        precision_val = precision[-1]
        recall_val = recall[-1]
        fscore_val = fscore[-1]
        
    else:
        # Build the CNN model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ CNN Model built.")
        
        # Train the model
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_data=(X_test, Y_test), shuffle=True, verbose=2)
        
        # Save model architecture and weights
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(Model_weights)
        
        # Save training history
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)
        
        # Predict on test set
        predict_prob = model.predict(X_test)
        predict = predict_prob.argmax(axis=1)
        calculateMetrics("CNN Model", predict, y_test)
        
        acc = hist.history['accuracy'][-1] * 100
        precision_val = precision[-1]
        recall_val = recall[-1]
        fscore_val = fscore[-1]
    
    return render(request, 'prediction.html', {
        'algorithm': 'CNN Model',
        'accuracy': round(acc, 2),
        'precision': round(precision_val, 2),
        'recall': round(recall_val, 2),
        'f1_score': round(fscore_val, 2),
    })

def CNN1_model(request):
    Model_file = os.path.join(model_folder, "RFC_Model.pkl")

    if os.path.exists(Model_file):
        rf_classifier = joblib.load(Model_file)
        predict = rf_classifier.predict(x_test)
        print("✅ Random Forest Classifier model loaded from file.")
        calculateMetrics("Random Forest Classifier", predict, y_test)
    else:
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train, y_train)
        joblib.dump(rf_classifier, Model_file)
        print("✅ Random Forest Classifier model trained and model weights saved.")
        predict = rf_classifier.predict(x_test)
        calculateMetrics("Random Forest Classifier", predict, y_test)

    latest_accuracy = accuracy[-1] if accuracy else 0
    latest_precision = precision[-1] if precision else 0
    latest_recall = recall[-1] if recall else 0
    latest_fscore = fscore[-1] if fscore else 0

    return render(request, 'prediction.html', {
        'algorithm': 'CNN',
        'accuracy': round(latest_accuracy, 2),
        'precision': round(latest_precision, 2),
        'recall': round(latest_recall, 2),
        'f1_score': round(latest_fscore, 2),
    })



labels=categories
#defining global variables to store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    
            
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from django.core.files.storage import default_storage
import joblib
import os

def predict_image_cnn(model, categories, image_path):
    # Load and preprocess the image
    img = imread(image_path)
    img_resized = resize(img, (64, 64, 3))  # Resize to 64x64x3
    img_flattened = [img_resized.flatten()]  # Flatten for RFC input

    # Predict the class
    output_number = model.predict(img_flattened)[0]
    output_name = categories[output_number]

    # Display image with prediction
    plt.imshow(img)
    plt.text(10, 10, f'Predicted Output of CNN: {output_name}', color='white',
             fontsize=12, weight='bold', backgroundcolor='black')
    plt.axis('off')
    plt.show()

    return output_name


def prediction_view(request):
    Test = True
    predicted_class = None

    Model_file = os.path.join(model_folder, "RFC_Model.pkl")
    categories = [
    'Bone/fractured', 'Bone/not fractured',
    'Brain/glioma', 'Brain/meningioma', 'Brain/notumor', 'Brain/pituitary',
    'Eye/diabetic_retinopathy', 'Eye/glaucoma', 'Eye/normal',
    'Lung/cancerous', 'Lung/non-cancerous',
    'Skin/BA- cellulitis', 'Skin/BA-impetigo',
    'Skin/FU-athlete-foot', 'Skin/FU-nail-fungus', 'Skin/FU-ringworm',
    'Skin/PA-cutaneous-larva-migrans',
    'Skin/VI-chickenpox', 'Skin/VI-shingles']
  # ✅ Update as needed

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)

        if os.path.exists(Model_file):
            # Load the trained Random Forest model
            cnn_classifier = joblib.load(Model_file)

            # Predict the class
            predicted_class = predict_image_cnn(cnn_classifier, categories, file_path)
        else:
            predicted_class = "Model not found."

        # Clean up uploaded file
        default_storage.delete(file_path)

    return render(request, 'prediction.html', {
        'test': Test,
        'predicted_class': predicted_class
    })


            




