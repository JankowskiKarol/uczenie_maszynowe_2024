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


def test(split, model, train_dataset, test_dataset, data_flag, BATCH_SIZE):
    
    # # Pobranie informacji o zbiorze danych na podstawie wybranej flagi
    info = INFO[data_flag]

    # # Pobranie informacji o zadaniu (task) na podstawie informacji o zbiorze danych
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
    
    # Wybór odpowiedniego dataloadera w zależności od wybranego podziału (train lub test)
    data_loader = train_loader_at_eval if split == 'train' else test_loader
    
    all_outputs = []

    # Wyłączenie obliczania gradientów podczas ewaluacji
    with torch.no_grad():
        # Pętla przez dane w dataloaderze
        for inputs, targets in data_loader:
            # Przepuszczenie danych przez model w celu uzyskania predykcji
            outputs = model(inputs, features=True)
            # Dodanie wyników do listy
            all_outputs.append(outputs)

    # Konkatenacja wyników z każdej iteracji
    all_outputs = torch.cat(all_outputs, dim=0)

        #     # Przygotowanie danych celu i wyników w zależności od zadania
        #     if task == 'multi-label, binary-class':
        #         targets = targets.to(torch.float32)  # Przekształcenie do float32 dla binarnej klasyfikacji wieloklasowej
        #         outputs = outputs.softmax(dim=-1)  # Normalizacja wyników modelu za pomocą softmax
        #     else:
        #         targets = targets.squeeze().long()  # Spreasowanie etykiet do postaci tensora liczbowego dla innych zadań
        #         outputs = outputs.softmax(dim=-1)  # Normalizacja wyników modelu za pomocą softmax
        #         targets = targets.float().resize_(len(targets), 1)  # Przekształcenie etykiet do postaci tensora
        
        #     # Dodanie rzeczywistych etykiet i wyników modelu do odpowiednich tensorów
        #     y_true = torch.cat((y_true, targets), 0)
        #     y_score = torch.cat((y_score, outputs), 0)

        # # Konwersja tensorów do numpy arrays
        # y_true = y_true.numpy()
        # y_score = y_score.detach().numpy()
        
        # # Inicjalizacja ewaluatora dla danego zbioru danych i wybranego podziału (train lub test)
        # evaluator = Evaluator(data_flag, split)
        
        # # Ewaluacja modelu za pomocą ewaluatora i uzyskanie metryk
        # metrics = evaluator.evaluate(y_score)
    
        # # Wypisanie wyników ewaluacji
        # # print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
        # result_text = f'{split} auc: {metrics[0]:.3f} acc: {metrics[1]:.3f}'
        # return result_text
    return all_outputs






class COPKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, init=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        # Initialization variables
        self.cluster_centers_ = None

        # Result variables
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self._initialize(X, const_mat)

        return self._fit(X, const_mat)

    def partial_fit(self, X, const_mat=None):
        if self.cluster_centers_ is None:
            self._initialize(X, const_mat)

        return self._fit(X, const_mat)

    def _initialize(self, X, const_mat=None):
        self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, const_mat=const_mat, random_state=self.random_state)

    def _fit(self, X, const_mat=None):
        tol = tolerance(X, self.tol)

        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self.assign_clusters(X, const_mat)

            if -1 in self.labels_:
                return self # exit - no solution

            # Estimate new centers
            prev_cluster_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])


            # Check for convergence
            if npl.norm(self.cluster_centers_ - prev_cluster_centers) < tol:
                break

        self.n_iter_ = iteration + 1

        return self

    def assign_clusters(self, X, const_mat):
        labels = np.full(X.shape[0], fill_value=-1)
        cdist = dist(X, self.cluster_centers_)

        for i in range(len(X)):
            for j in cdist[i].argsort():
                # check violate contraints
                for _ in labels[np.argwhere(const_mat[i] == 1)]:
                    if not (_ == j or j == -1):
                        continue

                if np.any(labels[np.argwhere(const_mat[i] == -1)] == j):
                    continue

                labels[i] = j
                break

        return labels
    



def get_part_labels(dataset):
    # Lista do przechowywania wszystkich etykiet
    part_labels = []

    # Iteracja przez cały zbiór danych i zbieranie etykiet z 1/5 danych 
    # derma
    # labeled_data_count = 100
    # unlabeled_data_count = 401
    # blood
    # labeled_data_count = 171
    # unlabeled_data_count = 684
    # path
    labeled_data_count = 359
    unlabeled_data_count = 1436
    total_count = labeled_data_count + unlabeled_data_count

    for i in range(0, len(dataset), total_count):
        # Dodanie etykietowanych danych
        for j in range(i, min(i + labeled_data_count, len(dataset))):
            _, label = dataset[j]
            part_labels.append(label.item())
        
        # Dodanie nieetykietowanych danych
        for j in range(i + labeled_data_count, min(i + total_count, len(dataset))):
            part_labels.append(-1)
    
    print(len(part_labels))
    return part_labels



def get_all_labels(dataset):
    # Lista do przechowywania wszystkich etykiet
    all_labels = []

    # Iteracja przez cały zbiór danych
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item())
    print(len(all_labels))


    return all_labels
    


# Funkcja do generowania macierzy ograniczeń
def create_constraint_matrix(labels):
    num_elements = len(labels)
    const_matrix = np.zeros((num_elements, num_elements))

    for i in range(num_elements):
        for j in range(num_elements):
            if labels[i] == -1 or labels[j] == -1:
                const_matrix[i][j] = 0  # Brak ograniczeń
            elif labels[i] == labels[j]:
                const_matrix[i][j] = 1  # Ta sama klasa
            else:
                const_matrix[i][j] = -1  # Różne klasy

    return const_matrix


# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred):
    # Obliczanie metryki accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy









# Przygotowanie danych

# data_ = 'dermamnist' 
# data_ = 'bloodmnist' 
data_ = 'pathmnist' 


train_dataset = prepare_train_dataset(data_)
test_dataset = prepare_test_dataset(data_)

# # Stworzenie modelu
model = ResNet(num_classes=len(INFO[data_]['label']))

# Trening modelu
model = train_ResNet(train_dataset, data_, BATCH_SIZE=128, NUM_EPOCHS=9, lr=0.001)

outputs = test('test', model,train_dataset, test_dataset, data_, BATCH_SIZE=64) 

# stworzenie listy zawierajacej wektory cech po 512 dla kazdego wektora na liscie, ilosć wektorów odpowiada iloscią próbek
flattened_data = outputs.reshape(outputs.shape[0], -1)
print("Test features shape:", flattened_data.shape)

flattened_data_numpy = flattened_data.detach().cpu().numpy()  # Konwersja tensora PyTorch na tablicę numpy


# macierz ograniczen
# const_matrix = np.zeros((len(flattened_data_numpy), len(flattened_data_numpy)))


# Wywołanie funkcji dla zbioru testowego
test_labels = get_part_labels(test_dataset)

all_labels = get_all_labels(test_dataset)


# Generowanie macierzy ograniczeń
const_matrix = create_constraint_matrix(test_labels)
# const_matrix = np.zeros((len(flattened_data_numpy), len(flattened_data_numpy)))

# # Wydrukowanie kształtu macierzy ograniczeń
# print("Kształt macierzy ograniczeń:", const_matrix.shape)



# Ustawienia NumPy do wyświetlania pełnej macierzy
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(const_matrix)

num_zeros = np.count_nonzero(const_matrix == 0)
num_ones = np.count_nonzero(const_matrix == 1)
num_neg_ones = np.count_nonzero(const_matrix == -1)

print(f"Liczba 0: {num_zeros}")
print(f"Liczba 1: {num_ones}")
print(f"Liczba -1: {num_neg_ones}")

# # Utwórzenie i dopasowanie model COPKMeans
copkmeans = COPKMeans(n_clusters=9)  # Utwórz instancję algorytmu COPKMeans   7,8,9
copkmeans.fit(flattened_data_numpy,const_matrix)  # Dopasuj model do danych



print(len(copkmeans.labels_))

# # Wyświetlenie centrów klastrów
# print("Centra klastrów:")
# print(copkmeans.cluster_centers_)

print(len(all_labels))

predicted_labels = copkmeans.labels_

# print("Predykcja")
# for label in predicted_labels:
#     print(label, end=" ")
# print("Poprawne") 
# for label in all_labels:
#     print(label, end=" ")

# Obliczenie dokładności
accuracy = calculate_metrics(all_labels, predicted_labels)

print(f"Accuracy: {accuracy}")

ari = adjusted_rand_score(all_labels, predicted_labels)
print(f"Adjusted Rand Index: {ari}")



# COPK .........................................................................................................

print("\nCOPKMeans bez ograniczen\n")


# Podział zbioru testowego na cztery części
X_test_splits = np.array_split(flattened_data_numpy, 4)
y_test_splits = np.array_split(all_labels, 4)
y_test_data = np.array_split(test_labels, 4)

# Generowanie macierzy ograniczeń dla każdego podzbioru i obliczanie metryk
for i in range(4):
    X_part = X_test_splits[i]
    y_part_true = y_test_splits[i]
    y_part_data = y_test_data[i]
    
    
    # Generowanie braku macierzy ograniczeń dla części zbioru testowego
    
    const_matrix_part = np.zeros((len(X_part), len(X_part)))
    
    
    # Dopasowanie modelu COPKMeans do części zbioru testowego
    copkmeans_part = COPKMeans(n_clusters=9)
    copkmeans_part.fit(X_part, const_matrix_part)
    
    y_part_pred = copkmeans_part.labels_
    
    # Wyświetlenie ilości elementów, etykiet wzorcowych oraz predykcyjnych dla części zbioru testowego
    print(f"Część {i+1}:")
    print(f"  Liczba elementów: {len(X_part)}")
    print(f"  Liczba etykiet wzorcowych: {len(y_part_true)}")
    print(f"  Liczba etykiet predykcyjnych: {len(y_part_pred)}")
    
    # Obliczenie metryk dla części zbioru testowego
    ari_part = adjusted_rand_score(y_part_true, y_part_pred)
    
    print(f"  Adjusted Rand Index: {ari_part}")


print("\nCOPKMeans z ograniczeniami\n")

# Podział zbioru testowego na cztery części
X_test_splits = np.array_split(flattened_data_numpy, 4)
y_test_splits = np.array_split(all_labels, 4)
y_test_data = np.array_split(test_labels, 4)

# Generowanie macierzy ograniczeń dla każdego podzbioru i obliczanie metryk
for i in range(4):
    X_part = X_test_splits[i]
    y_part_true = y_test_splits[i]
    y_part_data = y_test_data[i]
    
    
    # Generowanie macierzy ograniczeń dla części zbioru testowego
    const_matrix_part = create_constraint_matrix(y_part_data)
    
    num_zeros = np.count_nonzero(const_matrix_part == 0)
    num_ones = np.count_nonzero(const_matrix_part == 1)
    num_neg_ones = np.count_nonzero(const_matrix_part == -1)

    print(f"Liczba 0: {num_zeros}")
    print(f"Liczba 1: {num_ones}")
    print(f"Liczba -1: {num_neg_ones}")
    

    
    # Dopasowanie modelu COPKMeans do części zbioru testowego
    copkmeans_part = COPKMeans(n_clusters=9)
    copkmeans_part.fit(X_part, const_matrix_part)
    
    y_part_pred = copkmeans_part.labels_
    
    # Wyświetlenie ilości elementów, etykiet wzorcowych oraz predykcyjnych dla części zbioru testowego
    print(f"Część {i+1}:")
    print(f"  Liczba elementów: {len(X_part)}")
    print(f"  Liczba etykiet wzorcowych: {len(y_part_true)}")
    print(f"  Liczba etykiet predykcyjnych: {len(y_part_pred)}")
    
    # Obliczenie metryk dla części zbioru testowego
    ari_part = adjusted_rand_score(y_part_true, y_part_pred)
    
    print(f"  Adjusted Rand Index: {ari_part}")









class PCKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, init=None, weight=1, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.weight = weight
        self.random_state = random_state

        # Result variables
        self.cluster_centers_ = None
        self.labels_ = None
        self.n_iter_ = 0

    def fit(self, X, const_mat=None):
        self._initialize(X, const_mat)

        return self._fit(X, const_mat)

    def partial_fit(self, X, const_mat=None):
        if self.cluster_centers_ is None:
            self._initialize(X, const_mat)

        return self._fit(X, const_mat)

    def _initialize(self, X, const_mat=None):
        self.cluster_centers_ = initialize_centers(X, self.n_clusters, self.init, const_mat=const_mat, random_state=self.random_state)

    def _fit(self, X, const_mat=None):
        tol = tolerance(X, self.tol)

        for iteration in range(self.max_iter):
            # Assign clusters
            self.labels_ = self.assign_clusters(X, const_mat)

            # Estimate new centers
            prev_cluster_centers = self.cluster_centers_.copy()
            self.cluster_centers_ = np.array([
                X[self.labels_ == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            # Check for convergence
            if npl.norm(self.cluster_centers_ - prev_cluster_centers) < tol:
                break

        self.n_iter_ = iteration + 1

        return self

    def assign_clusters(self, X, const_mat):
        labels = np.full(X.shape[0], fill_value=-1)

        for i in range(len(X)):
            labels[i] = np.argmin([
                0.5 * np.sum(np.square(X[i] - self.cluster_centers_[j])) +
                np.sum(labels[np.argwhere(const_mat[i] == 1)] != j) * self.weight +
                np.sum(labels[np.argwhere(const_mat[i] == -1)] == j) * self.weight
                for j in range(self.n_clusters)
            ])

        return labels
    
    
    
    





# PCK .........................................................................................................

print("\nPCKMeans bez ograniczen\n")


# Podział zbioru testowego na cztery części
X_test_splits = np.array_split(flattened_data_numpy, 4)
y_test_splits = np.array_split(all_labels, 4)
y_test_data = np.array_split(test_labels, 4)

# Generowanie macierzy ograniczeń dla każdego podzbioru i obliczanie metryk
for i in range(4):
    X_part = X_test_splits[i]
    y_part_true = y_test_splits[i]
    y_part_data = y_test_data[i]
    
    
    # Generowanie braku macierzy ograniczeń dla części zbioru testowego
    
    const_matrix_part = np.zeros((len(X_part), len(X_part)))
    
    
    # Dopasowanie modelu pcPKMeans do części zbioru testowego
    pckkmeans_part = PCKMeans(n_clusters=9)
    pckkmeans_part.fit(X_part, const_matrix_part)
    
    y_part_pred = pckkmeans_part.labels_
    
    # Wyświetlenie ilości elementów, etykiet wzorcowych oraz predykcyjnych dla części zbioru testowego
    print(f"Część {i+1}:")
    print(f"  Liczba elementów: {len(X_part)}")
    print(f"  Liczba etykiet wzorcowych: {len(y_part_true)}")
    print(f"  Liczba etykiet predykcyjnych: {len(y_part_pred)}")
    
    # Obliczenie metryk dla części zbioru testowego
    ari_part = adjusted_rand_score(y_part_true, y_part_pred)
    
    print(f"  Adjusted Rand Index: {ari_part}")


print("\nPCKMeans z ograniczeniami\n")

# Podział zbioru testowego na cztery części
X_test_splits = np.array_split(flattened_data_numpy, 4)
y_test_splits = np.array_split(all_labels, 4)
y_test_data = np.array_split(test_labels, 4)

# Generowanie macierzy ograniczeń dla każdego podzbioru i obliczanie metryk
for i in range(4):
    X_part = X_test_splits[i]
    y_part_true = y_test_splits[i]
    y_part_data = y_test_data[i]
    
    
    # Generowanie macierzy ograniczeń dla części zbioru testowego
    const_matrix_part = create_constraint_matrix(y_part_data)
    
    num_zeros = np.count_nonzero(const_matrix_part == 0)
    num_ones = np.count_nonzero(const_matrix_part == 1)
    num_neg_ones = np.count_nonzero(const_matrix_part == -1)

    print(f"Liczba 0: {num_zeros}")
    print(f"Liczba 1: {num_ones}")
    print(f"Liczba -1: {num_neg_ones}")
    

    
    # Dopasowanie modelu pcKMeans do części zbioru testowego
    pckkmeans_part = PCKMeans(n_clusters=9)
    pckkmeans_part.fit(X_part, const_matrix_part)
    
    y_part_pred = pckkmeans_part.labels_
    
    # Wyświetlenie ilości elementów, etykiet wzorcowych oraz predykcyjnych dla części zbioru testowego
    print(f"Część {i+1}:")
    print(f"  Liczba elementów: {len(X_part)}")
    print(f"  Liczba etykiet wzorcowych: {len(y_part_true)}")
    print(f"  Liczba etykiet predykcyjnych: {len(y_part_pred)}")
    
    # Obliczenie metryk dla części zbioru testowego
    ari_part = adjusted_rand_score(y_part_true, y_part_pred)
   
    print(f"  Adjusted Rand Index: {ari_part}")














