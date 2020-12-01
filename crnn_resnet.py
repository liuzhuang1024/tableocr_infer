import torch.nn as nn
import torch.nn.functional as F
import torch



import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 4, 3])
        #self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [2, 2, 6, 4])

    def forward(self, input):
        return self.ConvNet(input)

class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 8), int(output_channel / 4), int(output_channel / 2), output_channel,256]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0,0))
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
            0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=(0,0))
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
            1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 0))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
            2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))
        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
            3], kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 0))
        self.conv5_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
            4], kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False)
        self.bn5_1 = nn.BatchNorm2d(self.output_channel_block[4])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.maxpool4(x)
        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)

        x = self.maxpool5(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)

        
        #x = self.conv4_2(x)
        #x = self.bn4_2(x)
        #x = self.relu(x)
        return x


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn_l = nn.LSTM(nIn, nHidden, bidirectional=False)
        self.rnn_r = nn.LSTM(nIn, nHidden, bidirectional=False)

    def forward(self, input):
        input_revers = torch.flip(input,[0])
        recurrent, _ = self.rnn_l(input)
        recurrent_revers,_ = self.rnn_r(input_revers)
        recurrent_revers = torch.flip(recurrent_revers,[0])
        output = recurrent + recurrent_revers

        return output

class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=True):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.cnn = ResNet_FeatureExtractor(1, 512)
        self.rnn_1 = BidirectionalLSTM(nh,nh,nh)
        self.rnn_2 = BidirectionalLSTM(nh,nh,nh)
        self.fc = nn.Linear(nh,nclass)
        '''
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        '''
    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        conv = conv.permute(0,1,3,2)
        b, c, h, w = conv.size()
        conv = conv.reshape(b,c,1,h*w)
        b, c, h, w = conv.size()
        # print(conv.shape)
        #print("b, c, h, w",b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #output = F.log_softmax(self.rnn(conv), dim=2)
        rnn_1 = self.rnn_1(conv) + conv
        rnn_2 = self.rnn_2(rnn_1) + conv
        T,n,h = rnn_2.size()
        t_rec = rnn_2.view(T*b,h)
        output = self.fc(t_rec)
        output = output.view(T,b,-1)
        #print("output",output.shape)
        output = F.softmax(output, dim=2)
        output = output.permute(1 , 0 , 2)
        #print("inside",output.shape)
        return output
