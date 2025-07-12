# Arabic Letter Recognition with ResNet-50

This project implements an Arabic handwritten letter recognition system using a fine-tuned **ResNet‑50** convolutional neural network. It is designed to work on grayscale images (32x32), resized to match ResNet‑50 input requirements.

The model is trained and evaluated using a dataset of Arabic letter images, with filenames encoding their true labels (e.g., `id_1234_label_5.png`). This project includes training, evaluation, visualization, and export of test predictions as CSV.

---

## 🚀 Features

- ✅ **Transfer Learning** using pretrained ResNet-50 from ImageNet  
- 🎯 Fine-tuned on 28 Arabic letter classes  
- 🧠 Custom `Dataset` loader for labeled PNG images  
- 📊 Per-class accuracy, confusion matrix, and visualizations  
- ⏹️ Early stopping and learning rate scheduling  
- 📁 Compatible with Google Colab (with Google Drive integration)

---

## 📦 Technologies Used

- Python 3.12
- PyTorch
- torchvision
- scikit-learn
- pandas, matplotlib, seaborn

---

## 📁 Dataset Structure

Expected structure after extraction:

Computer Vision/

├── Train Images 13440x32x32/

│ └── train/

└── Test Images 3360x32x32/

└── test/



Each image file should follow the format:  
`id_XXXX_label_YY.png` → where `YY` is the class label (1-based index from 1 to 28).

---

## 🧪 Running the Code

### On Google Colab:
1. Upload the `Computer Vision.zip` file to your Google Drive.
2. Update the `ZIP_PATH` in the script with the correct path in Drive.
3. Run `arabic_resnet50.py` — it will:
   - Mount your Drive
   - Extract the dataset
   - Load and fine-tune ResNet‑50
   - Save predictions in `test_predictions.csv`

### Locally:
1. Unzip your dataset in the expected structure.
2. Run the script with Python 3.x.

---

## 📈 Results

- ✅ **Training Accuracy:** ~99.3%  
- ✅ **Test Accuracy:** ~91.3%  
- Visual outputs:
  - Confusion matrix
  - Distribution of predictions
  - Per-class accuracy bar chart

---

## 📤 Output

- `test_predictions.csv`:  
  Contains test image names and their predicted labels.

---

## ✍️ Authors

- [Abdelrahman Hussien](https://github.com/abdo-hussien)
- [Mohamed Hussien](https://github.com/mohameddhussien)

