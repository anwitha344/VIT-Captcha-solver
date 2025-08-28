# Captcha Recognition with CNNs

link to download the model: https://drive.google.com/file/d/1zH0xwTzztHUNy-k3pbE-0q56q5vQh6xg/view?usp=drive_link

This project is an end-to-end pipeline I built for solving captchas using Convolutional Neural Networks (CNNs).  
I handled every stage: scraping captchas from VIT’s VTOP portal, labeling them, preprocessing the dataset, building a multi-output CNN, and training it with regularization.

---

## Project Overview
- **Goal**: Build a working captcha recognizer for 6-character captchas from VTOP.  
- **Stack**: Python, Requests, Pandas, NumPy, TensorFlow/Keras, Matplotlib.  
- **Dataset**: ~1050 labeled captchas.  
- **Model**: Custom CNN with 6 classification heads.  

---

## Dataset

The dataset is stored in a CSV file with the following fields:

* **base64 string** → encoded captcha image.
* **label** → text solution (e.g., `"A7X9PQ"`).
* **type** → `T1` or `T2`, depending on noise style.

For training, I decoded the base64 strings, converted the images to grayscale, resized them, and prepared both one-hot encoded labels and raw text labels.

---

# Full Documentation of project process

### Step 1: Captcha Collection

I started by inspecting VTOP’s captcha system with Chrome DevTools. The images were being loaded as **base64 strings** from a backend endpoint.
Initially, direct requests using requests lkibrary in python failed since the server required authentication. To fix this, I used `requests.session()`to visit the login page and first set the session cookies, and with those in place, I was able to request the captcha endpoint.

I then wrote a script to fetch the base64 strings repeatedly and store them into a CSV file, building up the dataset.

---

### Step 2: Labeling Captchas

To make labeling efficient, I created a small UI using tkinter in Google Colab.

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

To separate them, I wrote a script that examined the border pixel RGB values. If they were white, It marked the captcha as `T1`; if grey, as `T2`. This classification was stored in an additional CSV column.

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
┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_3       │ (None, 100, 100,  │          0 │ -                 │
│ (InputLayer)        │ 1)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (Conv2D)   │ (None, 100, 100,  │        320 │ input_layer_3[0]… │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_9     │ (None, 50, 50,    │          0 │ conv2d_9[0][0]    │
│ (MaxPooling2D)      │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_12          │ (None, 50, 50,    │          0 │ max_pooling2d_9[… │
│ (Dropout)           │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (Conv2D)  │ (None, 50, 50,    │     18,496 │ dropout_12[0][0]  │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_10    │ (None, 25, 25,    │          0 │ conv2d_10[0][0]   │
│ (MaxPooling2D)      │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_13          │ (None, 25, 25,    │          0 │ max_pooling2d_10… │
│ (Dropout)           │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (Conv2D)  │ (None, 25, 25,    │     73,856 │ dropout_13[0][0]  │
│                     │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_11    │ (None, 12, 12,    │          0 │ conv2d_11[0][0]   │
│ (MaxPooling2D)      │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_14          │ (None, 12, 12,    │          0 │ max_pooling2d_11… │
│ (Dropout)           │ 128)              │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ flatten_3 (Flatten) │ (None, 18432)     │          0 │ dropout_14[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_3 (Dense)     │ (None, 512)       │  9,437,696 │ flatten_3[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_15          │ (None, 512)       │          0 │ dense_3[0][0]     │
│ (Dropout)           │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_0 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_1 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_2 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_3 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_4 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ char_5 (Dense)      │ (None, 34)        │     17,442 │ dropout_15[0][0]  │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 9,635,020 (36.75 MB)
 Trainable params: 9,635,020 (36.75 MB)
 Non-trainable params: 0 (0.00 B)

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
### Training history and graphs
<img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/17578d36-b66e-472d-b0cb-eb1c4e39f6a2" />

<img width="1018" height="855" alt="image" src="https://github.com/user-attachments/assets/eb7624d7-1e47-4f36-b4cf-be2edcc554e4" />



---
### Further trained model on examples it gets wront by filtering data set 
<img width="988" height="547" alt="image" src="https://github.com/user-attachments/assets/0fcc7b61-e28b-40c1-b1a1-f5f1717f6b61" />

Dues to small data set and little low accuracy I compared my model's accuracy to other pretrained models- ResNet18 and InceptionV3 after fine tuning and added layers and other functions to compare accuracy 

#### Results

| Model       | Accuracy per character | whole captcha accuracy |
| ----------- | -----------------------| ---------------------- |
| Custom CNN  | 98%                    | 58%                    |
| ResNet18    | 27%                    | 00%                    |
| InceptionV3 | 31%                    | 5%                     |

#### Hence the problem is with data size 
---

## Future Work

* Collect more data to increase accuracy.
* Explore YOLO approach
* Experiment with transformer-based OCR.

---

## Acknowledgements
Humans I know for labelling 1000+ captchas

## My goal and how you can contribute

The model was able to achieve around 97–98% accuracy for individual characters, but the overall CAPTCHA prediction accuracy was ~60%, primarily due to the limited dataset size.

My next steps is to try YOLO approach and simultainiusly collect a much much larger dataset, build a UI, and deploy it using Streamlit so it can be accessed via a shareable link

If you’d like to be part of the project and contribute, please fill out the Google Form with your contact details. I’ll share the Streamlit link with you once it’s live so you can help with CAPTCHA labeling.

https://docs.google.com/forms/d/e/1FAIpQLSfQe2tXsPMo6Zqrm1S8BFvLTLfuqbvwjt8v1KhDu6lh_Vj82w/viewform?usp=header

