import torch
import torch.nn as nn 
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class MnistDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file, header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()
        pass
    
    pass

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        
        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.counter = 0;
        self.progress = []

        pass
    
    
    def forward(self, inputs):
        return self.model(inputs)
    
    
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        
        loss = self.loss_function(outputs, targets)

        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass
    
    pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0;
        self.progress = []       

        pass

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):
        g_output = self.forward(inputs)

        d_output = D.forward(g_output)

        loss = D.loss_function(d_output, targets)

        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass
    
    pass


def generate_random_seed(size):
    #@ randn      표준정규분포 난수; 평균 0 표준편차 1
    #@ rand         표준정규분포 난수; 0 ~ 1 균일분포
    #@ randint 
    random_data = torch.randn(size)
    return random_data

    
if __name__ == '__main__':
    mnist_dataset = MnistDataset('_data/mnist_train.csv')
    
    G = Generator()
    D = Discriminator()

    epochs = 1

    import time
    st = time.time()

    for epoch in range(epochs):
        print ("epoch = ", epoch + 1)
        for label, image_data_tensor, target_tensor in mnist_dataset:
            D.train(image_data_tensor, torch.FloatTensor([1.0]))
            D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
            
            G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

            pass

    ed = time.time() - st
    print(round(ed/60, 2))

    D.plot_progress()
    G.plot_progress()


# Run G
    f, axarr = plt.subplots(2,3, figsize=(16,8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(100))
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            pass
        pass
    plt.show()


    seed1 = generate_random_seed(100)
    out1 = G.forward(seed1)
    img1 = out1.detach().numpy().reshape(28,28)
    plt.imshow(img1, interpolation='none', cmap='Blues')
    plt.show()

    seed2 = generate_random_seed(100)
    out2 = G.forward(seed2)
    img2 = out2.detach().numpy().reshape(28,28)
    plt.imshow(img2, interpolation='none', cmap='Blues')
    plt.show()

    count = 0

    # seed1 to seed2
    f, axarr = plt.subplots(3,4, figsize=(16,8))
    for i in range(3):
        for j in range(4):
            seed = seed1 + (seed2 - seed1)/11 * count
            output = G.forward(seed)
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            count = count + 1
            pass
        pass
    plt.show()

    # sum of seeds
    seed3 = seed1 + seed2
    out3 = G.forward(seed3)
    img3 = out3.detach().numpy().reshape(28,28)
    plt.imshow(img3, interpolation='none', cmap='Blues')
    plt.show()

    # sub of seeds
    seed4 = seed1 - seed2
    out4 = G.forward(seed4)
    img4 = out4.detach().numpy().reshape(28,28)
    plt.imshow(img4, interpolation='none', cmap='Blues')
    plt.show()

    # mul of seeds
    seed4 = seed1 * seed2
    out4 = G.forward(seed4)
    img4 = out4.detach().numpy().reshape(28,28)
    plt.imshow(img4, interpolation='none', cmap='Blues')
    plt.show()

