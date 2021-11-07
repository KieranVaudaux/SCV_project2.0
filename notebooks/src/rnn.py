import torch
import utils
import numpy as np
from torch import nn
from torch.nn import functional as F
import pandas as pd


class RecNet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_output):
        super().__init__()
        self.fc_x2h = nn.Linear(dim_input, dim_recurrent)
        self.fc_h2h = nn.Linear(dim_recurrent, dim_recurrent)
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output)

    def forward(self, input_):
        h = input_.new_zeros(input_.size(0), self.fc_h2y.weight.size(1))
        for t in range(input_.size(1)):
            h = F.relu(self.fc_x2h(input_) + self.fc_h2h(h))
        return self.fc_h2y(h)


if __name__ == "__main__":
    model = RecNet(dim_input=8, dim_recurrent=60, dim_output=1)

    df = utils.get_cointrin(remove_dubious=True)

    # drop the "quality" and "souid" columns
    df.drop(list(df.filter(regex="Q_")), axis=1, inplace=True)
    df.drop(list(df.filter(regex="SOUID")), axis=1, inplace=True)

    # Drop columns with too many nans
    df.drop(columns=["PP", "SD"], inplace=True)

    # Drop the few remaining rows with nans
    df.dropna(inplace=True)

    train_iters, lr = 2000, 1e-3
    window_size = 10

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for k in range(train_iters):
        window = df.loc[k * window_size : (k + 1) * window_size, :]
        # window = df.sample(window_size)
        input_ = torch.from_numpy(window.drop("TG", axis=1).to_numpy()).float()
        target = torch.from_numpy(window.TG.to_numpy()).float()
        output = model(input_)
        loss = mse_loss(output, target)
        losses.append(loss.detach().data)
        # print("output: {}, target: {}".format(output, target))
        print("iter: {}, loss: {:.2f}".format(k, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = pd.Series(losses)
    print("med: {}, std: {}".format(losses.median(), losses.std()))
