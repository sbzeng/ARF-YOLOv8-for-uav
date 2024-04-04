class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
        
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class Conv2(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = h_swish()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


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

