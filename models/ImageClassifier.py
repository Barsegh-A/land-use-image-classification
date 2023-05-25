import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        
        # Define your model architecture layers here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, 21)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x
    
    def backpropagation(self, optimizer, criterion, inputs, labels):
        optimizer.zero_grad()  # Zero the gradients
        outputs = self.forward(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters
    
    def optimize(self, train_loader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()  # Define the loss criterion
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)  # Define the optimizer
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.backpropagation(optimizer, criterion, inputs, labels)
                running_loss += criterion(self.forward(inputs), labels).item()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
