from torch import nn

dropout = 0.2
in_features = 28 * 28


class TFC(nn.Sequential):
    def __init__(self, hidden1, hidden2, hidden3):
        super(TFC, self).__init__(
            nn.Dropout(p=dropout),

            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(hidden3, 10)
        )

    def forward(self, x):
        x = x.reshape((-1, in_features))
        return super(TFC, self).forward(x)
