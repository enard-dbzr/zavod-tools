from torch import nn


class RegressedLSTM(nn.Module):
    def __init__(self, input_dim=26, output_dim=8, context_size=20, layers=1, regressor=None):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, context_size, batch_first=True, num_layers=layers)
        if regressor is None:
            self.regressor = nn.Linear(context_size, output_dim)
        else:
            self.regressor = regressor

    def forward(self, x):
        return self.regressor(self.lstm(x)[0])
