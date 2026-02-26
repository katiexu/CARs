import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import pathlib
import logging
from utils.metrics import AverageMeter
from tqdm import tqdm
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
import numpy as np
from Arguments import Arguments

# Qiskit imports
from math import pi
import os

def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]    
    arch_code = ([i for i in range(2, n_qubits+1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1,0)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code) 

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # number of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design

def single_enta_to_design(single, enta, arch_code, fold=1):
    """
    Generate a design list usable by QNET from single and enta codes

    Args:
        single: Single-qubit gate encoding, format: [[qubit, gate_config_layer0, gate_config_layer1, ...], ...]
                Each two bits of gate_config represent a layer: 00=Identity, 01=U3, 10=data, 11=data+U3
        enta: Two-qubit gate encoding, format: [[qubit, target_layer0, target_layer1, ...], ...]
              Each value represents the target qubit position in that layer
        arch_code_fold: [n_qubits, n_layers]

    Returns:
        design: List containing quantum circuit design info, each element is (gate_type, [wire_indices], layer)
    """
    design = []
    single = qubit_fold(single, 0, fold)
    enta = qubit_fold(enta, 1, fold)

    n_qubits, n_layers = arch_code

    # Process each layer
    for layer in range(n_layers):
        # First process single-qubit gates
        for qubit_config in single:
            qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The config for each layer is at position: 1 + layer*2 and 1 + layer*2 + 1
            config_start_idx = 1 + layer * 2
            if config_start_idx + 1 < len(qubit_config):
                gate_config = f"{qubit_config[config_start_idx]}{qubit_config[config_start_idx + 1]}"

                if gate_config == '01':  # U3
                    design.append(('U3', [qubit], layer))
                elif gate_config == '10':  # data
                    design.append(('data', [qubit], layer))
                elif gate_config == '11':  # data+U3
                    design.append(('data', [qubit], layer))
                    design.append(('U3', [qubit], layer))
                # 00 (Identity) skip

        # Then process two-qubit gates
        for qubit_config in enta:
            control_qubit = qubit_config[0] - 1  # Convert to 0-based index
            # The target qubit position in the list: 1 + layer
            target_idx = 1 + layer
            if target_idx < len(qubit_config):
                target_qubit = qubit_config[target_idx] - 1  # Convert to 0-based index

                # If control and target qubits are different, add C(U3) gate
                if control_qubit != target_qubit:
                    design.append(('C(U3)', [control_qubit, target_qubit], layer))
                # If same, skip (equivalent to Identity)

    return design

def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]    
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)
    return arch.transpose(1, 0)

def shift_ith_element_right(original_list, i):
    """
    对列表中每个item的第i个元素进行循环右移一位
    
    Args:
        original_list: 原始列表，如 [[3, 0, 5], [4, 3, 6], [5, 1, 7], [1, 2, 8]]
        i: 要循环右移的元素索引，如 i=1 表示第二个元素
   
    """   
    ith_elements = [item[i] for item in original_list]    
    # 循环右移一位：最后一个元素移到开头
    shifted_ith = [ith_elements[-1]] + ith_elements[:-1]    
    result = [item[:i] + [shifted_ith[idx]] + item[i+1:] for idx, item in enumerate(original_list)]
    return result

def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:            
            if phase == 0:
                q = job[0]
                job_list += [[fold*(q-1)+1+i] + job[1:] for i in range(0, fold)]
            else:
                job = [i-1 for i in job]
                q = job[0]
                indices = [i for i, x in enumerate(job) if x < q]
                enta = [[fold*j+i+1 for j in job] for i in range(0,fold)]
                for i in indices:
                    enta = shift_ith_element_right(enta, i)
                job_list += enta
    else:
        job_list = jobs
    return job_list

class TQLayer_old(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits

        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(10)]

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3)) # each U3 gate needs 3 parameters
            self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3)) # each CU3 gate needs 3 parameters
        rot_trainable = True
        enta_trainable = True

        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):

                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                     self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer]))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer]))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def data_uploading(self, qubit):
        input = [
        {"input_idx": [0], "func": "ry", "wires": [qubit]},
        {"input_idx": [1], "func": "rz", "wires": [qubit]},
        {"input_idx": [2], "func": "rx", "wires": [qubit]},
        {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x, n_qubits=4, task_name=None):
        bsz = x.shape[0]
        if task_name.startswith('QML'):
            x = x.view(bsz, n_qubits, -1)
        else:
            kernel_size = self.args.kernel
            x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
            if kernel_size == 4:
                x = x.view(bsz, 6, 6)
                tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
                x = tmp.reshape(bsz, -1, 10).transpose(1,2)
            else:
                x = x.view(bsz, 4, 4).transpose(1,2)


        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)


        for layer in range(self.design['n_layers']):
            for j in range(self.n_wires):
                if self.design['qubit_{}'.format(j)][0][layer] != 0:
                    self.uploading[j](qdev, x[:,j])
                if self.design['qubit_{}'.format(j)][1][layer] == 0:
                    self.rots[j + layer * self.n_wires](qdev, wires=j)

            for j in range(self.n_wires):
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        out = self.measure(qdev)
        if task_name.startswith('QML'):
            out = out[:, :2]    # only take the first two measurements for binary classification

        return out

class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits        
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(self.n_wires)]

        self.q_params_rot = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each U3 gate needs 3 parameters
        self.q_params_enta = nn.Parameter(pi * torch.rand(self.args.n_layers, self.args.n_qubits, 3))  # each CU3 gate needs 3 parameters
        
        self.measure = tq.MeasureAll(tq.PauliZ)


    def data_uploading(self, qubit):
        input = [
            {"input_idx": [0], "func": "ry", "wires": [qubit]},
            {"input_idx": [1], "func": "rz", "wires": [qubit]},
            {"input_idx": [2], "func": "rx", "wires": [qubit]},
            {"input_idx": [3], "func": "ry", "wires": [qubit]},
        ]
        return input

    def forward(self, x):
        bsz=x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        
        for i in range(len(self.design)):
            if self.design[i][0] == 'U3':                
                layer = self.design[i][2]
                qubit = self.design[i][1][0]
                params = self.q_params_rot[layer][qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.u3(qdev, wires=self.design[i][1], params=params)
            elif self.design[i][0] == 'C(U3)':               
                layer = self.design[i][2]
                control_qubit = self.design[i][1][0]
                params = self.q_params_enta[layer][control_qubit].unsqueeze(0)  # 重塑为 [1, 3]
                tqf.cu3(qdev, wires=self.design[i][1], params=params)
            else:   # data uploading: if self.design[i][0] == 'data'
                j = int(self.design[i][1][0])
                self.uploading[j](qdev, x[:,j])
        out = self.measure(qdev)
        return out


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.QuantumLayer = TQLayer(self.args, self.design)
        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features=4, out_features=10)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        bsz = x.shape[0]
        x = self.adaptive_pool(x)
        x = x.view(bsz, 4, 4)

        x = self.QuantumLayer(x)
        x = self.fc(x)  # 映射为 (n,10)
        return x
    def input_to_representation(self, x):
        bsz = x.shape[0]
        x = self.adaptive_pool(x)
        x = x.view(bsz, 4, 4)

        x = self.QuantumLayer(x)
        return x

    def representation_to_output(self, h):
        return self.fc(h)

    def get_hooked_modules(self) -> dict[str, nn.Module]:
        return {
            "adaptive_pool": self.adaptive_pool,
            "QuantumLayer": self.QuantumLayer,
            "fc": self.fc
        }
    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        """
        One epoch of the training loop
        Args:
            device: device where tensor manipulations are done
            dataloader: training set dataloader
            optimizer: training optimizer

        Returns:
            average loss on the training set
        """
        self.train()
        train_loss = []
        loss_meter = AverageMeter("Loss")
        train_bar = tqdm(dataloader, unit="batch", leave=False)
        for x_batch, label_batch in train_bar:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            pred_batch = self.forward(x_batch)
            loss = self.criterion(pred_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), len(x_batch))
            train_bar.set_description(f"Training Loss {loss_meter.avg:.3g}")
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)
    def test_epoch(
        self, device: torch.device, dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        One epoch of the testing loop
        Args:
            device: device where tensor manipulations are done
            dataloader: test set dataloader

        Returns:
            average loss and accuracy on the training set
        """
        self.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for x_batch, label_batch in dataloader:
                x_batch = x_batch.to(device)
                label_batch = label_batch.to(device)
                pred_batch = self.forward(x_batch)
                loss = self.criterion(pred_batch, label_batch)
                test_loss.append(loss.cpu().numpy())
                test_acc.append(
                    torch.count_nonzero(label_batch == torch.argmax(pred_batch, dim=-1))
                    .cpu()
                    .numpy()
                    / len(label_batch)
                )

        return np.mean(test_loss), np.mean(test_acc)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        lr: int = 1e-03,
        n_epoch: int = 50,
        patience: int = 10,
        checkpoint_interval: int = -1,
    ) -> None:
        """
        Fit the classifier on the training set
        Args:
            device: device where tensor manipulations are done
            train_loader: training set dataloader
            test_loader: test set dataloader
            save_dir: path where checkpoints and model should be saved
            lr: learning rate
            n_epoch: maximum number of epochs
            patience: optimizer patience
            checkpoint_interval: number of epochs between each save

        Returns:

        """
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_acc = 0
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss, test_acc = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train Loss {train_loss:.3g} \t "
                f"Test Loss {test_loss:.3g} \t"
                f"Test Accuracy {test_acc * 100:.3g}% \t "
            )
            if test_acc <= best_test_acc:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_acc = test_acc.data
                waiting_epoch = 0
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info(f"Early stopping activated")
                break

    def save(self, directory: pathlib.Path) -> None:
        """
        Save a model.
        Parameters
        ----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        path_to_model = directory / ("vqc_model.pt")
        torch.save(self.state_dict(), path_to_model)


if __name__ == "__main__":
    import torch
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")  # 屏蔽无关警告

