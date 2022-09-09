from torch import nn


class TFC(nn.Sequential):
    def __init__(self, hidden1, hidden2, hidden3):
        super(TFC, self).__init__(
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 10)
        )

    def forward(self, input):
        input = input.reshape((-1, 28 * 28))
        return super(TFC, self).forward(input)
