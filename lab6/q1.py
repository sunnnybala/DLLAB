import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# question1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(device)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN()
torch.save(model,'mnist.pt')
model = torch.load('mnist.pt',weights_only=False)
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


accuracy = evaluate_model(model, test_loader)
print(f"Test accuracy on FashionMNIST: {accuracy * 100:.2f}%")


# def fine_tune_model(model, train_loader, test_loader, epochs=5):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#         accuracy = evaluate_model(model, test_loader)
#         print(
#             f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")
#
#
# fine_tune_model(model, train_loader, test_loader, epochs=5)
#
#
# def show_predictions(model, test_loader, num_images=5):
#     model.eval()
#     data_iter = iter(test_loader)
#     inputs, labels = next(data_iter)
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs, 1)
#
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
#     for i in range(num_images):
#         axes[i].imshow(inputs[i].squeeze(), cmap='gray')
#         axes[i].set_title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
#         axes[i].axis('off')
#
#     plt.show()


# show_predictions(model, test_loader)
