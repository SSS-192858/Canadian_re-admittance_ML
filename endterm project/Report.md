# ML Assignment 2 Report - OBSIDIAN 
## Canadian Hospital Readmittance Challenge

1. Siddharth Kothari (IMT2021019)
2. Sankalp Kothari (IMT2021028)
3. M Srinivasan (IMT2021058)


## Overview

This report contains the development of a hospital readmission prediction model using SVMs and neural network (implemented in PyTorch). The goal is to predict whether a patient will be readmitted to the hospital based on various features related to their health and medical history.

The best model that we could come up was with Neural Networks(with comparision to SVMs and Neural Networks).

We had used the same pre-processing as done for the previous assignment for the second part of the assignment. 

We had used Neural Network with Adam optimizer for the same and the results of the prediction were same as that of RandomForests / Xgboost.

### Dataset Description -
The instances represent the hospitalized patients as recorded in the hospitals in Canada. Each row has about 50 columns corresponding to it. Details include the age range, weight, gender, race, etc. The description of some of the important columns are mentioned in detail below:-

* **age**: Age of the person grouped in 10-year intervals - [0, 10), [10, 20),…, [90, 100)
* **admission_type_id**: Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available.
* **number_outpatient/inpatient**: Number of outpatient or inpatient visits of the patient respectively in the year preceding the encounter.
* **number_diagnoses**: Number of diagnoses entered to the system.
* **Days to inpatient readmission**: 0, if the patient was readmitted in less than 30 days, 1 if the patient was readmitted in more than 30 days, and 2 for no record of readmission.

Along with the above columns there are columns that correspond to each individual medicines. The have the following as categorical data:-
The feature indicates whether the drug was prescribed or there was a change in the dosage. Values: up if the dosage was increased during the encounter, down if the dosage was decreased, steady if the dosage did not change, and no if the drug was not prescribed.


## SVM

### Model Architecture and Hyperparatmeters
The SVM model and it's hyperparameters is as follows:

1) We have used the `rbf` kernel for our SVM and the hyperparameters such as `C`, `gamma` are required.
2) `C` is the regularization parameter, controlling the trade-off between achieving a smooth decicion boundary and classifying training points correctly.
3) `gamma` - This is the Kernel coefficient for `rbf` kernel, controlling the shape of the boundary.

* We used Optuna to get the best hyper parameters and they are stored in the `best_params` dictionary(code below).

### Model fitting
```python
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Train SVM using the best hyperparameters
    best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf', random_state=42)
    best_svm.fit(X_train, y_train)

```

### Accuracy - 
The best accuracy we got with the `rbf` kernel and the hyper-parameters that we got is 0.652.



## Neural Networks

### Model Architecture
The neural network architecture is as follows:

1) `Input Layer`: Number of neurons equal to the number of features in the dataset.
2) `Hidden Layers`: Four hidden layers with decreasing numbers of neurons (configurable).
3) `Output Layer`: Three neurons corresponding to the classes.


### Model fitting
The following were the steps we took for fitting the neural network model:-
1) We created a class with 4 hidden layers, 1 output layer and 1 input layer, the code for the same is highlighted below:
```python
    class ANN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        hidden_dim_3: int,
        hidden_dim_4: int,
        n_classes:int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim_1),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim_1, out_features=hidden_dim_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim_2),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim_2, out_features=hidden_dim_3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim_3),
            nn.Dropout(dropout),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=hidden_dim_3, out_features=hidden_dim_4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim_4),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(in_features=hidden_dim_4, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x (torch.Tensor): (batch_size, in_dim) the input

            Output:
                (torch.Tensor): (batch_size, n_classes) the output
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x
```
2) We then defined a custom `Data` class that extends the `Dataset` class of `PyTorch`.It is responsible for representing our dataset and organizing its features and labels.
```python
    class Data(Dataset):
    def __init__(
        self,
        data
    ):
        n = data.shape[1]
        self.features = torch.tensor(data.iloc[:, 0:n-1].values.astype(np.int64), dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, -1].values.astype(np.int64), dtype=torch.int64)
    # def _build(self):
        # scaler = MinMaxScaler(feature_range=())
        # scaler = StandardScaler()

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)
```
3) We then did the train, test split with 80:20 ratio respectively. We kept the batch size as 512.
4) We the used the data passed through custom `Data` class to `DataLoader` class. This shall take care of the loading and handling the pre-processed data into the neural networks.
5) We set the number of iterations(`epochs`) to be 5. 
6) We use `Adam` optimizer and use `Cross Entorpy Loss` as our loss function for training our model.
7) We get the output and the pass it through `Softmax` layer to the correst probabilites of each class and take the max of each set of output and that's the prediction of that set.

### Accuracy and Hyperparamter Tuning
Training parameters such as learning rate, dropout, and layer dimensions are optimized using Optuna.
The following hyperparameters are optimized:
1) Number of hidden layers
2) Neurons in each hidden layer
3) Dropout rate
4) Learning rate

The best accuracy with tuned hyper-parameters we got was: 0.7111173498034812


## Conclusion 

The neural networks and other models such as RandomForests and XGboost(ensemble methods) give very similar accuracy for this dataset.

We would prefer using the latter ones as the training time required to do them is much lesser than the former(neural networks).

The SVMs took around 3-4 hours to train and run, but the accuracy had no better results than RandomForests/XGboost.
Even the Neural Networks took longer (definitely not as long as SVM), but gave the same results.