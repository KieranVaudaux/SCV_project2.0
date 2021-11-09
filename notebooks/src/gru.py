import torch
import utils
import numpy as np
from torch import nn
from torch.nn import functional as F
import pandas as pd

class Gru(nn.Module):
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):
        super().__init__()
        self.gru = nn.GRU(input_size = dim_input,
            hidden_size = dim_recurrent,
            num_layers = num_layers)
        self.fc_y = nn.Linear(dim_recurrent, dim_output)
    def forward(self, input):
        # Get the last layer's last time step activation
        output, _ = self.gru(input)
        output = output[-1]
        return self.fc_y(F.relu(output))

if __name__ == "__main__":
    dim_recs = [30, 40, 50]
    lrs = [1e-3, 1e-2, 1e-1]
    window_size = 4
    train_iters = 400

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
            for _ in range(0, 10):
                model = Gru(dim_input=9, dim_recurrent=dim, num_layers= 1, dim_output=1)

                mse_loss = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                losses = []

                for k in range(train_iters):
                    window = df.loc[k * window_size : (k + 1) * window_size, :]
                    # window = df.sample(window_size)
                    input_ = torch.from_numpy(window.to_numpy()).float()
                    input_ = input_.view(1,input_.shape[0], input_.shape[1])
                    # print(input_.shape)
                    target = torch.from_numpy(
                        df.loc[
                            (k + 1) * window_size + 1 : (k + 2) * window_size + 1, :
                        ].TG.to_numpy()
                    ).view(-1,1).float()
                    output = model(input_)

                    # print("output: {}, target: {}".format(output, target))

                    # print(
                    #     "input: {}, output: {}, target: {}".format(
                    #         input_.shape, output.shape, target.shape
                    #     )
                    # )
                    loss = mse_loss(output, target)
                    losses.append(loss.detach().data)
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
                print(
                    "[dim: {}, lr: {}, window: {}] avg_median: {:.2e}, avg_mad: {:.2e}".format(
                        dim, lr, window_size, medians.mean(), mads.mean()
                    )
                )
