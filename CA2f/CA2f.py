class CA2f(nn.Module):
    def __init__(self, inp, reduction=32):
        self.oup = inp
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = Conv2(int(inp*0.5), mip, k=1, s=1)

        self.conv_h = nn.Conv2d(mip, int(inp*0.5), kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, int(inp*0.5), kernel_size=1, stride=1, padding=0)

        self.conv2 = Conv(inp, self.oup, k=1, s=1)
        
    def forward(self, x):
        x1 = x[:, :int(self.oup*0.5) , :, :]
        identity = x1
        n, c, h, w = x1.size()
        x_h = self.pool_h(x1)
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        
        x2 = x[:, int(self.oup*0.5):, :, :]
       
        return self.conv2(torch.cat([out, x2], dim=1))

