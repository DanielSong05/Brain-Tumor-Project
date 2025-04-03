# Import necessary libraries
import pickle             # For serializing and deserializing Python objects (loading saved data)
from torch.utils.data import Dataset  # Base class for creating custom datasets in PyTorch
from PIL import Image     # For handling image operations (e.g., opening, converting images)
import numpy as np        # For working with arrays and numerical operations

# Define a custom dataset class for brain tumor images by extending PyTorch's Dataset class.
class BrainTumorDataset(Dataset):
    # The __init__ method is the constructor which initializes the dataset object.
    def __init__(self, pickle_file, transform=None):
        """
        Args:
            pickle_file (str): Path to the pickle file containing the dataset.
                                 The pickle file should have serialized data.
            transform (callable, optional): Optional transform to be applied on a sample.
                                            This is often used for data augmentation or normalization.
        """
        # Open the pickle file in read-binary mode ('rb') and load the data.
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Store the transformation function (if provided) for later use.
        self.transform = transform

    # The __len__ method returns the total number of samples in the dataset.
    def __len__(self):
        return len(self.data)
    
    # The __getitem__ method retrieves and processes a sample from the dataset at the given index.
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to be fetched.
        Returns:
            tuple: (image, label) where image is the processed image and label is the class label.
        """
        image, label = self.data[index]

        # Convert NumPy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Adjust label indexing (assuming labels are 1-indexed in pickle)
        label = label - 1
        
        return image, label
