from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

import medmnist
from medmnist import INFO, Evaluator


import numpy as np
from numpy import linalg as npl

from common import initialize_centers, tolerance
from scipy.spatial.distance import cdist as dist


from common import tolerance, initialize_centers, INIT_NEIGHBORHOOD
from warnings import warn
from time import time

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import confusion_matrix


from sklearn.metrics import accuracy_score


def prepare_train_dataset(data_flag):
    # Flaga do pobierania danych - ustawiona na True
    download = True

    # Pobranie informacji o zbiorze danych na podstawie wybranej flagi
    info = INFO[data_flag]

    # Pobranie informacji o zadaniu (task), liczbie kanałów (n_channels)
    # oraz liczbie klas (n_classes) na podstawie informacji o zbiorze danych
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # Pobranie klasy danych na podstawie nazwy klasy w module 'medmnist'
    DataClass = getattr(medmnist, info['python_class'])

    # Przygotowanie transformacji dla preprocessingu danych
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # Zamiana danych na tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizacja danych
    ])

    # Tworzenie zbiorów treningowego i testowego z wykorzystaniem odpowiednich transformacji i ewentualnie pobieraniem danych
    train_dataset = DataClass(split='train', transform=data_transform, download=download)

    # print("TRAIN_DATASET\n")
    # print(train_dataset)
    # print("\n")

    return train_dataset

def prepare_test_dataset(data_flag):
    # Flaga do pobierania danych - ustawiona na True
    download = True

    # Pobranie informacji o zbiorze danych na podstawie wybranej flagi
    info = INFO[data_flag]

    # Pobranie informacji o zadaniu (task), liczbie kanałów (n_channels)
    # oraz liczbie klas (n_classes) na podstawie informacji o zbiorze danych
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # Pobranie klasy danych na podstawie nazwy klasy w module 'medmnist'
    DataClass = getattr(medmnist, info['python_class'])

    # Przygotowanie transformacji dla preprocessingu danych
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # Zamiana danych na tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizacja danych
    ])

    # Tworzenie zbiorów treningowego i testowego z wykorzystaniem odpowiednich transformacji i ewentualnie pobieraniem danych
    # train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    
     # Wydrukowanie ilości elementów w zbiorze testowym
    print("Number of samples in test dataset:", len(test_dataset))
    

    return test_dataset


# Definicja bloku rezydualnego
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or x.size(1) != out.size(1):
            identity = nn.Conv2d(x.size(1), out.size(1), kernel_size=1, stride=self.stride, bias=False)(identity)
            identity = nn.BatchNorm2d(out.size(1))(identity)

        out += identity
        out = self.relu(out)

        return out

# Definicja modelu ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes,features=False):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # if features==False:
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if features==False:
            x = self.fc(x)

        return x


def train_ResNet(train_dataset, data_flag, BATCH_SIZE, NUM_EPOCHS, lr):
    # Flaga do pobierania danych - ustawiona na True
    download = True
    # Pobranie informacji o zbiorze danych na podstawie wybranej flagi
    info = INFO[data_flag]

    # Pobranie informacji o zadaniu (task), liczbie kanałów (n_channels)
    # oraz liczbie klas (n_classes) na podstawie informacji o zbiorze danych
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    model = ResNet(num_classes=n_classes)

    # Tworzenie dataloadera dla zbioru treningowego:
    # Dataloader to narzędzie pozwalające na efektywne ładowanie danych w małych partiach (mini-batche) podczas treningu.
    # Parametr 'dataset' wskazuje na zbiór danych, z którego będą pobierane przykłady do treningu.
    # Parametr 'batch_size' określa, ile przykładów jest przetwarzanych naraz podczas jednego kroku treningowego.
    # Parametr 'shuffle=True' oznacza, że w każdej epoce kolejność przykładów jest losowo zmieniana,
    # co pomaga uniknąć ukierunkowania modelu na konkretną sekwencję przykładów.
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
        
    # Pętla treningowa - dla każdej epoki treningowej
    for epoch in range(NUM_EPOCHS):
        # Inicjalizacja zmiennych przechowujących liczbę poprawnych predykcji oraz całkowitą liczbę przykładów
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
    
        # Ustawienie modelu w trybie treningu - wymagane dla modułów takich jak Dropout czy BatchNorm
        model.train()
       
        # Pętla przez dane treningowe przy użyciu tqdm w celu śledzenia postępu
        for inputs, targets in tqdm(train_loader):
            # Wyzerowanie gradientów optymalizatora
            optimizer.zero_grad()
        
            # Przepuszczenie danych przez model w celu uzyskania predykcji
            outputs = model(inputs)
        
            # Przygotowanie danych celu w zależności od zadania
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)  # Przekształcenie do float32 dla binarnej klasyfikacji wieloklasowej
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()  # Spreasowanie etykiet do postaci tensora liczbowego dla innych zadań
                loss = criterion(outputs, targets)
        
            # Propagacja wsteczna i aktualizacja wag modelu
            loss.backward()
            optimizer.step()
    return model

from sklearn.metrics import precision_score, recall_score



from sklearn.metrics import precision_score, recall_score



def test(split, model, train_dataset, test_dataset, data_flag, BATCH_SIZE):
    
    # Pobranie informacji o zbiorze danych na podstawie wybranej flagi
    info = INFO[data_flag]

    # Pobranie informacji o zadaniu (task) na podstawie informacji o zbiorze danych
    task = info['task']
    
    # Wydrukowanie ilości elementów w zbiorze testowym
    print("Number of samples in test dataset:", len(test_dataset))

    # Tworzenie dataloadera dla zbioru treningowego przy ewaluacji:
    # Ten dataloader ma podobną konstrukcję jak poprzedni, ale będzie używany do ewaluacji modelu, więc
    # dane nie są przemieszane (shuffle=False).
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    # Tworzenie dataloadera dla zbioru testowego:
    # Podobnie jak wcześniej, ale dane testowe nie są przemieszane.
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    
    # Ustawienie modelu w trybie ewaluacji
    model.eval()
    
    # Inicjalizacja pustych tensorów dla prawdziwych etykiet i wyników modelu
    y_true = torch.tensor([])  # Tensor przechowujący rzeczywiste etykiety
    y_score = torch.tensor([])  # Tensor przechowujący wyniki modelu
    
    # Inicjalizacja list do przechowywania wyników precision i recall dla każdej klasy
    precision_list = []
    recall_list = []

    # Wyłączenie obliczania gradientów podczas ewaluacji
    with torch.no_grad():
        # Pętla przez dane w dataloaderze
        for inputs, targets in train_loader_at_eval if split == 'train' else test_loader:
            
            # Przepuszczenie danych przez model w celu uzyskania predykcji
            outputs = model(inputs)
  

            # Przygotowanie danych celu i wyników w zależności od zadania
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)  # Przekształcenie do float32 dla binarnej klasyfikacji wieloklasowej
                outputs = outputs.softmax(dim=-1)  # Normalizacja wyników modelu za pomocą softmax
            else:
                targets = targets.squeeze().long()  # Spreasowanie etykiet do postaci tensora liczbowego dla innych zadań
                outputs = outputs.softmax(dim=-1)  # Normalizacja wyników modelu za pomocą softmax
                targets = targets.float().resize_(len(targets), 1)  # Przekształcenie etykiet do postaci tensora
        
            # Dodanie rzeczywistych etykiet i wyników modelu do odpowiednich tensorów
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

            # Obliczenie precision i recall dla każdej klasy
            precision = precision_score(targets, outputs.argmax(dim=1), average='weighted', zero_division=0)
            recall = recall_score(targets, outputs.argmax(dim=1), average='weighted', zero_division=0)
            precision_list.append(precision)
            recall_list.append(recall)

        # Konwersja tensorów do numpy arrays
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
        
        # Obliczenie średniego precision i recall dla wszystkich klas
        avg_precision = sum(precision_list) / len(precision_list)
        avg_recall = sum(recall_list) / len(recall_list)
        
        print ( f'{split} acc: {metrics[1]:.3f} precision: {avg_precision:.3f} recall: {avg_recall:.3f}')
      
        









# Przygotowanie danych

data_ = 'dermamnist' 
# data_ = 'bloodmnist' 
# data_ = 'pathmnist' 


train_dataset = prepare_train_dataset(data_)
test_dataset = prepare_test_dataset(data_)

# # Stworzenie modelu
model = ResNet(num_classes=len(INFO[data_]['label']))

# Trening modelu
model = train_ResNet(train_dataset, data_, BATCH_SIZE=128, NUM_EPOCHS=9, lr=0.001)

outputs = test('test', model,train_dataset, test_dataset, data_, BATCH_SIZE=64) 
