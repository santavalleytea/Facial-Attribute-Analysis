import torch
import torch.nn as nn
import torch.optim as optim
from data_load import train_loader, val_loader, test_loader

class FacialAtributeCNN(nn.Module):
    def __init__(self):
        super(FacialAtributeCNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 40)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(-1, 128 * 16 * 16)

        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

# Instantiate model
model = FacialAtributeCNN()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train() # Model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        validate_model(model, val_loader, criterion)

def validate_model(model, val_loader, criterion):
    model.eval()  # Model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss / len(val_loader)}")

# Function to evaluate the model on the test dataset
def evaluate_model(model, test_loader):
    model.eval()  # Model to evaluation mode
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary
            total += labels.size(0) * labels.size(1)  # Total number of attributes
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Main function to train and evaluate the model
if __name__ == '__main__':
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
    
    # Evaluate the model
    evaluate_model(model, test_loader)