import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import BrainTumorDataset
from model import SimpleCNN
from config import TRAIN_PICKLE, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_CLASSES, IMG_SIZE

def train():
    # Define transforms for training (RGB, normalized)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load the full dataset from the pickle file
    full_dataset = BrainTumorDataset(TRAIN_PICKLE, transform=transform)
    
    # Split the dataset: 80% for training, 20% for testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    # Create a DataLoader for the training subset
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, loss function, optimizer
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == labels).sum().item()
            epoch_correct += correct
            epoch_total += images.size(0)
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Batch [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        
        epoch_accuracy = (epoch_correct / epoch_total) * 100
        print(f'Epoch [{epoch + 1}/{EPOCHS}] Training Accuracy: {epoch_accuracy:.2f}%')
    
    # Save the trained model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Finished training and saved model to cnn_model.pth")

if __name__ == '__main__':
    train()
