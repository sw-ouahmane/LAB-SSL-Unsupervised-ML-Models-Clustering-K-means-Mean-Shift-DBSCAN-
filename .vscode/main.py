
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random 
import matplotlib.pyplot as plt 


class ContrastiveMNIST(Dataset):
    
    def __init__(self, mnist_dataset):
        self.data = mnist_dataset

  
    def __getitem__(self, index):
        img1, label1 = self.data[index]
        should_match = random.randint(0, 1)

        if should_match:
           
            while True:
                img2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 == label2:
                    break
        else:
           
            while True:
                img2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                if label1 != label2:
                    break
                 
        return img1, img2, torch.tensor([int(label1 == label2)], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

  
    def forward_once(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):

        return self.forward_once(x1), self.forward_once(x2)




class ContrastiveLoss(nn.Module):
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)


        loss = label * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()


transform = transforms.Compose([transforms.ToTensor()])
train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = ContrastiveMNIST(train_mnist)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5):
    total_loss = 0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")





def show_pair(img1, img2, distance, label):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1.squeeze(), cmap='gray')
    ax[1].imshow(img2.squeeze(), cmap='gray')
    plt.suptitle(f"Distance: {distance:.2f} - {'Same' if label else 'Different'}")
    plt.show()

test_img1, test_img2, test_label = train_dataset[0]
with torch.no_grad():
    e1, e2 = model(test_img1.unsqueeze(0).to(device), test_img2.unsqueeze(0).to(device))
    dist = F.pairwise_distance(e1, e2).item()
    show_pair(test_img1, test_img2, dist, test_label.item())
