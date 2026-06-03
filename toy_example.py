import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path
from utils.dataset import generate_mnist_concept_dataset
from models.mnist import ClassifierMnist
from experiments.mnist import concept_to_class
from explanations.concept import CAR
from explanations.feature import CARFeatureImportance

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = Path.cwd()/"data/mnist"

# Load model
model = ClassifierMnist(latent_dim=42).to(device)

# Load concept sets
X_train, C_train = generate_mnist_concept_dataset(concept_to_class["Loop"],
                                               data_dir,
                                               train=True,
                                               subset_size=200,
                                               random_seed=42)
X_test, C_test = generate_mnist_concept_dataset(concept_to_class["Loop"],
                                               data_dir,
                                               train=False,
                                               subset_size=100,
                                               random_seed=42)

# Evaluate latent representation
X_train = torch.from_numpy(X_train).to(device)
X_test = torch.from_numpy(X_test).to(device)
H_train = model.input_to_representation(X_train)
H_test = model.input_to_representation(X_test)

# Fit CAR classifier
car = CAR(device)
car.fit(H_train.detach().cpu().numpy(), C_train)
car.tune_kernel_width(H_train.detach().cpu().numpy(), C_train)

# Predict concept labels
C_predict = car.predict(H_test.detach().cpu().numpy())

# Load test set
test_set = MNIST(data_dir, train=False, download=True)
test_set.transform = transforms.Compose([transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# Evaluate feature importance on the test set
baselines = torch.zeros([1, 1, 28, 28]) # Black image baseline
car_features = CARFeatureImportance("Integrated Gradient", car, model, device)
feature_importance = car_features.attribute(test_loader, baselines=baselines)