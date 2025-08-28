# Captcha Recognition with CNNs

This project is an end-to-end pipeline I built for solving captchas using Convolutional Neural Networks (CNNs).  
I handled every stage: scraping captchas from VIT’s VTOP portal, labeling them, preprocessing the dataset, building a multi-output CNN, and training it with regularization.

---

## Project Overview
- **Goal**: Build a working captcha recognizer for 6-character captchas from VTOP.  
- **Stack**: Python, Requests, Pandas, NumPy, TensorFlow/Keras, Matplotlib.  
- **Dataset**: ~1050 labeled captchas.  
- **Model**: Custom CNN with 6 classification heads.  

---

##Installation

git clone <repo-link>
cd <repo>
pip install -r requirements.txt

---

## Usage

### Train the Model

python train.py

### Predict on a Captcha

python predict.py --image path/to/captcha.png

---

## 📂 Dataset

The dataset is stored in a CSV file with the following fields:

* **base64 string** → encoded captcha image.
* **label** → text solution (e.g., `"A7X9PQ"`).
* **type** → `T1` or `T2`, depending on noise style.

For training, I decoded the base64 strings, converted the images to grayscale, resized them, and prepared both one-hot encoded labels and raw text labels.

---

# 📖 Full Documentation of project process

### Step 1: Captcha Collection

I started by inspecting VTOP’s captcha system with Chrome DevTools. The images were being loaded as **base64 strings** from a backend endpoint.
Initially, direct requests failed since the server required authentication. To fix this, I used `requests.session()`. Visiting the login page first set the session cookies, and with those in place, I was able to request the captcha endpoint.

I then wrote a script to fetch the base64 strings repeatedly and store them into a CSV file, building up the dataset.

---

### Step 2: Labeling Captchas

To make labeling efficient, I created a small UI in Google Colab.

* It allowed selecting how many captchas to label in one session.
* Captchas were displayed in a large, clear format.
* A progress bar tracked completion.
* Users could save and exit at any point.

I crowdsourced the labeling to humans(my friends and family)

---

### Step 3: Captcha Type Segregation

While reviewing the dataset, I noticed two distinct captcha styles:

* **Type 1**: grey stroke noise.
* **Type 2**: checkered grey background.

To separate them, I wrote a script that examined the border pixel RGB values. If they were white, I marked the captcha as `T1`; if grey, as `T2`. This classification was stored in an additional CSV column.

<img width="1280" height="516" alt="image" src="https://github.com/user-attachments/assets/dc33250b-b008-48a8-aff3-2c80ed894620" />


---

### Step 4: Preprocessing

For training, I decoded the base64 strings into NumPy arrays, converted them to grayscale, and resized them consistently.
The dataset was organized as:

* **images** → captcha images.
* **y** → one-hot encoded labels.
* **labels** → raw text strings.

I split the dataset with `train_test_split`, keeping labels aligned:

* 90% training, 10% testing.
* `random_state=42` for reproducibility.

Since each captcha contains 6 characters, I split the labels into 6 arrays, one per character position. The CNN could then predict each character separately.

---

### Step 5: Model Architecture

The model processes an input of shape `(50, 200, 1)` (grayscale captcha).

* **Feature Extraction**: multiple Conv2D + ReLU + MaxPooling layers.
* **Dense Layers**: flattened convolutional output passed through fully connected layers.
* **Outputs**: six softmax branches, one per character. Each branch predicts from all possible classes (A–Z, 0–9).


Input → Conv → Conv → Pool → Conv → Conv → Pool → Dense → [6 outputs]


### Step 6: Training

I trained the model with:

* Optimizer: **Adam**
* Loss: **Categorical Cross-Entropy** (multi-output)
* Batch Size: **32**
* Epochs: **50**

To improve generalization:

* Added **Dropout** and **L2 regularization**.
* Used a **learning rate scheduler** to reduce LR when validation accuracy stopped improving.

---

## Results

| Model       | Accuracy |
| ----------- | -------- |
| Custom CNN  | XX%      |
| ResNet18    | XX%      |
| InceptionV3 | XX%      |

---

## Future Work

* Collect more data to increase accuracy.
* Explore YOLO 
* Experiment with transformer-based OCR.

---

## Acknowledgements
Humans I know for labelling 1000+ captchas

