import torch


class CNN3D(torch.nn.Module):

    def __init__(self, ch_in=3, dim_out=1, init_n_feat=32):
        super(CNN3D, self).__init__()
        n_feat = init_n_feat
        self.encoder1 = self._block(ch_in, n_feat)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(n_feat, n_feat*2)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(n_feat*2, n_feat*4)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = self._block(n_feat*4, n_feat*8)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self._block(n_feat*8, n_feat*16)
        self.output = torch.nn.Linear(n_feat*16, dim_out)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        return self.output(torch.mean(bottleneck,dim=[2,3,4]))

    @staticmethod
    def _block(ch_in, n_feat):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=ch_in, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(num_features=n_feat),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm3d(num_features=n_feat),
            torch.nn.ReLU(inplace=True))
