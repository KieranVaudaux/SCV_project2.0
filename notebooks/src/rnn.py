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
    dim_recs = [30, 40, 50]
    lrs = [1e-3, 1e-2, 1e-1]
    window_size = 35
    train_iters = 600

    df = utils.get_cointrin(remove_dubious=True)

    # drop the "quality" and "souid" columns
    df.drop(list(df.filter(regex="Q_")), axis=1, inplace=True)
    df.drop(list(df.filter(regex="SOUID")), axis=1, inplace=True)

    # Drop columns with too many nans
    df.drop(columns=["PP", "SD"], inplace=True)

    # Drop the few remaining rows with nans
    df.dropna(inplace=True)

    for dim in dim_recs:
        for lr in lrs:
            medians = []
            mads = []
            for _ in range(0,10):
                model = RecNet(dim_input=9, dim_recurrent=dim, dim_output=9)

                mse_loss = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                losses = []

                for k in range(train_iters):
                    window = df.loc[k * window_size : (k + 1) * window_size, :]
                    # window = df.sample(window_size)
                    input_ = torch.from_numpy(window.to_numpy()).float()
                    target = torch.from_numpy(df.loc[(k + 1) * window_size + 1, :].to_numpy()).view(-1, 1).float()
                    output = model(input_)
                    loss = mse_loss(output, target)
                    losses.append(loss.detach().data)
                    # print("output: {}, target: {}".format(output.data, target))
                    # print("iter: {}, loss: {:.2f}".format(k, loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses = pd.Series(losses)
                medians.append(losses.median())
                mads.append((losses - losses.median()).median())
            else:
                medians = np.array(medians)
                mads = np.array(mads)
                print("[dim: {}, lr: {}, window: {}] avg_median: {:.2e}, avg_mad: {:.2e}".format(dim, lr, window_size, medians.mean(), mads.mean()))
