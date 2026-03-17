import torch.nn as nn
import torch.nn.functional as F


def get_model_function(args):
    n_outputs = args.output  # 可以是 int 或 [int, int]
    dropout_prob = args.dropout
    if args.network == 'CNN7':
        input_channel = args.input_channel
        return CNN7(input_channel, n_outputs, dropout_prob)
    elif args.network == 'MLPs':
        hidden_layers = args.hidden_dim
        n_inputs = args.input_features
        return MLPs(n_inputs, n_outputs, hidden_layers, dropout_prob)


class MLPs(nn.Module):
    def __init__(self, inputs_units, outputs_units, hidden_layers, dropout_prob=0.0):
        super(MLPs, self).__init__()

        # 创建隐藏层
        layer_list = nn.ModuleList()
        for hidden_units in hidden_layers:
            layer_list.append(nn.Linear(inputs_units, hidden_units))
            layer_list.append(nn.BatchNorm1d(hidden_units))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout_prob))
            inputs_units = hidden_units

        self.hidden_layers = nn.Sequential(*layer_list)

        # 解析输出维度
        if isinstance(outputs_units, (list, tuple)) and len(outputs_units) == 2:
            out1, out2 = outputs_units[0], outputs_units[1]
        else:
            out1 = out2 = outputs_units

        # 定义两个输出头
        self.output_layer1 = nn.Linear(hidden_layers[-1], out1)

    def forward(self, x):
        h = self.hidden_layers(x.float())

        return self.output_layer1(h)



class CNN7(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.0):
        super(CNN7, self).__init__()

        self.dropout_rate = dropout_rate

        # 卷积层保持不变
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)

        # BN 层保持不变
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(128)

        # 解析输出维度
        if isinstance(n_outputs, (list, tuple)) and len(n_outputs) == 2:
            out1, out2 = n_outputs[0], n_outputs[1]
        else:
            out1 = out2 = n_outputs

        # 输出层根据解析结果设置
        self.l_c1 = nn.Linear(128, out1)
        self.l_c2 = nn.Linear(128, out2)

    def forward(self, x):
        h = self.c1(x)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.size()[3])
        h = h.view(h.size(0), -1)

        return  self.l_c1(h)


def call_bn(bn, x):
    return bn(x)