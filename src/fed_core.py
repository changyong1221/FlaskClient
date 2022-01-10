import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise


class BaseModel:
    def __init__(self, net):
        self.net = net
        self.history = None
        self.weights = None
        self.loss_func = None
        self.optimizer = None
        self.clip = 12      # 裁剪系数
        self.q = 0.05
        self.eps = 16
        self.delta = 1e-4
        self.tot_T = 10
        self.E = 100
        self.sigma = compute_noise(1, self.q, self.eps, self.E*self.tot_T, self.delta, 1e-5)      # 高斯分布系数

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def load_model(self, file_path, weight=False):
        if weight:
            self.weights = torch.load(file_path)
            self.net.load_state_dict(self.weights, strict=True)
        else:
            self.net = torch.load(file_path)

    def set_model_settings(self, loss_func, optimizer):
        self.loss_func = loss_func
        self.optimizer = optimizer

    def train(self, train_data, train_labels, epoches, batch_size):
        train_dl = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
        train_loss = 0
        num = 0
        for epoch in range(epoches):
            for data, label in train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_func(preds, label)

                clipped_grads = {name: torch.zeros_like(param) for name, param in self.net.named_parameters()}
                self.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip)
                for name, param in self.net.named_parameters():
                    clipped_grads[name] += param.grad
                self.net.zero_grad()
                # add gaussian noise
                for name, param in self.net.named_parameters():
                    clipped_grads[name] += torch.normal(0, self.sigma*self.clip, clipped_grads[name].shape).to(self.dev)
                for name, param in self.net.named_parameters():
                    param.grad = clipped_grads[name]

                self.optimizer.step()
                self.optimizer.zero_grad()
                if epoch == epoches - 1:
                    num += 1
                    train_loss += float(loss.item())
        return train_loss / num

    def save_model(self, save_path, weight=False):
        if weight:
            torch.save(self.net.state_dict(), save_path)
        else:
            torch.save(self.net, save_path)

    def evaluate(self, test_data, test_labels, batch_size):
        test_data_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            sum_accu = 0
            num = 0
            for data, label in test_data_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            acc = sum_accu / num
        return acc.item()


class FedClient(BaseModel):
    def __init__(self, net=None, ID=0):
        super().__init__(net)
        self.ID = ID

    def setJob(self, jobAdress):
        pass

    def upload(self, weight=False):
        pass


class FedServer(BaseModel):
    def __init__(self, net=None):
        super().__init__(net)
        self.clients_weights_sum = None
        self.global_weights = None
        self.clients_num = 0

    def load_client_weights(self, model_path_list):
        num_clients = 0
        for model_path in model_path_list:
            num_clients += 1
            cur_parameters = torch.load(model_path)
            if self.clients_weights_sum is None:
                self.clients_weights_sum = {}
                for key, var in cur_parameters.items():
                    self.clients_weights_sum[key] = var.clone()
            else:
                for var in cur_parameters:
                    self.clients_weights_sum[var] = self.clients_weights_sum[var] + cur_parameters[var]
        self.clients_num = num_clients

    def load_client_models(self, model_path_list):
        pass
        # arr = []
        # for model_path in model_path_list:
        #     c_model = load_model(model_path)
        #     arr.append(c_model.get_weights())
        # self.models_data = np.array(arr)

    def fed_avg(self):
        # FL average
        self.global_weights = {}
        for var in self.clients_weights_sum:
            self.global_weights[var] = (self.clients_weights_sum[var] / self.clients_num)
        self.net.load_state_dict(self.global_weights, strict=True)

    def setJob(self, jobAdress):
        pass

    def get_modelsID_list_from_block(self):
        pass


if __name__ == '__main__':
    pass