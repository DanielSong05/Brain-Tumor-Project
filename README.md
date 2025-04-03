# Brain Tumor Classification with CNN

This project uses a custom Convolutional Neural Network (CNN) built with **PyTorch** to classify brain MRI images into one of three tumor types:
- Glioma
- Meningioma
- Pituitary

The dataset is preprocessed into a `.pickle` file, while the raw images are stored separately. The model is trained on 80% of the data and evaluated on the remaining 20%, achieving high accuracy on unseen images.

---

## Features

- **Custom CNN Architecture:** Built with 3 convolutional layers and 2 fully connected layers.
- **RGB Image Processing:** Utilizes torchvision transforms for resizing, normalizing, and converting images to tensors.
- **Dataset Management:** Uses a pickle file to store filename-label tuples with raw images kept in a separate directory.
- **Robust Evaluation:** Implements an 80/20 train-test split using PyTorch’s `random_split`.
- **Modular Design:** All code is organized under the `src/` directory.
- **Future Extension:** Plans to build a web application (using Flask or FastAPI) to allow users to upload images and display predictions in real time.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier/src

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Usage

### Train the model
```bash
python main.py
```
This command will:
- Train the CNN on 80% of the dataset.
- Evaluate the model on the remaining 20%.
- Save the trained model to cnn_model.pth.

## Future Plans
I plan to extend this project into a fully functional web application that will allow users to:
- Upload a brain MRI image.
- Automatically classify the tumor type.
- Display the prediction and confidence score in real time.

## Dependencies
- torch
- torchvision
- Pillow
- numpy
- matplotlib
- flask (for future use)

## Results
Achieved 94.5% test accuracy using this architecture and pipeline. Further improvements can be made through hyperparameter tuning or transfer learning approaches.



