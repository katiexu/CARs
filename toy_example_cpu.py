import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from pathlib import Path
from utils.dataset import generate_mnist_concept_dataset
from models.mnist import ClassifierMnist
from experiments.mnist import concept_to_class
from explanations.concept import CAR
from explanations.feature import CARFeatureImportance

# 修改这里：强制使用CPU
device = torch.device("cpu")
print(f"Using device: {device}")

data_dir = Path.cwd()/"data/mnist"

# Load model - 修改：加载到CPU
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
# 修改：使用CPU张量
X_train = torch.from_numpy(X_train).float().to(device)  # 添加.float()确保数据类型
X_test = torch.from_numpy(X_test).float().to(device)
H_train = model.input_to_representation(X_train)
H_test = model.input_to_representation(X_test)

# Fit CAR classifier
# 修改：CAR也使用CPU
car = CAR(device)  # device已经是'cpu'
car.fit(H_train.detach().cpu().numpy(), C_train)
car.tune_kernel_width(H_train.detach().cpu().numpy(), C_train)

# Predict concept labels
C_predict = car.predict(H_test.detach().cpu().numpy())

# Load test set
test_set = MNIST(data_dir, train=False, download=True)
test_set.transform = transforms.Compose([transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# Evaluate feature importance on the test set
# 修改：baselines也使用CPU
baselines = torch.zeros([1, 1, 28, 28])  # 默认就在CPU上
car_features = CARFeatureImportance("Integrated Gradient", car, model, device)
feature_importance = car_features.attribute(test_loader, baselines=baselines)