import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import BrainTumorDataset
from config import TRAIN_PICKLE, BATCH_SIZE, IMG_SIZE, NUM_CLASSES
from model import SimpleCNN

def evaluate_model():
    # Define transforms for testing (should match training transforms)
    transforms_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load the full dataset from the pickle file
    full_dataset = BrainTumorDataset(TRAIN_PICKLE, transform=transforms_test)
    
    # Split the dataset: 80% for training, 20% for testing.
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    # We only need the test split here, so discard the training split
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    # Create a DataLoader for the test subset
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load the model and its trained weights
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load('cnn_model.pth'))
    
    # Set device and switch model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss_history = []
    test_accuracy_history = []
    
    # Evaluate the model on the test dataset.
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss_history.append(loss.item())
            
            # Compute accuracy using argmax
            predicted = torch.argmax(y_pred, dim=1)
            correct = (predicted == y).sum().item()
            batch_accuracy = (correct / x.size(0)) * 100
            test_accuracy_history.append(batch_accuracy)
    
    # Compute average test loss and accuracy.
    avg_test_loss = sum(test_loss_history) / len(test_loss_history)
    avg_test_accuracy = sum(test_accuracy_history) / len(test_accuracy_history)
    
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%')

if __name__ == '__main__':
    evaluate_model()
