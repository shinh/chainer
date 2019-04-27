import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class BasicUnit(chainer.Chain):

    def __init__(self, channels):
        super(BasicUnit, self).__init__()

        initialW = initializers.HeNormal()
        ch = channels // 2

        with self.init_scope():
            self.conv1 = L.Convolution2D(ch, ch, 1,
                                         initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch, 3, pad=1, groups=ch,
                                         initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(ch, ch, 1,
                                         initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(ch)

    def forward(self, x):
        l, h = F.split_axis(x, 2, axis=1)
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.bn2(self.conv2(h))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.concat((l, h), axis=1)
        return h


class DownSampleUnit(chainer.Chain):

    def __init__(self, in_channels, out_channels):
        super(DownSampleUnit, self).__init__()

        initialW = initializers.HeNormal()
        ch = out_channels // 2

        with self.init_scope():
            self.rconv1 = L.Convolution2D(in_channels, ch, 1,
                                          initialW=initialW, nobias=True)
            self.rbn1 = L.BatchNormalization(ch)
            self.rconv2 = L.Convolution2D(ch, ch, 3,
                                          stride=2, pad=1, groups=ch,
                                          initialW=initialW, nobias=True)
            self.rbn2 = L.BatchNormalization(ch)
            self.rconv3 = L.Convolution2D(ch, ch, 1,
                                          initialW=initialW, nobias=True)
            self.rbn3 = L.BatchNormalization(ch)

            self.lconv1 = L.Convolution2D(in_channels, in_channels, 3,
                                          stride=2, pad=1, groups=in_channels,
                                          initialW=initialW, nobias=True)
            self.lbn1 = L.BatchNormalization(in_channels)
            self.lconv2 = L.Convolution2D(in_channels, ch, 1,
                                          initialW=initialW, nobias=True)
            self.lbn2 = L.BatchNormalization(ch)

    def forward(self, x):
        l = r = x
        l = self.lbn1(self.lconv1(l))
        l = F.relu(self.lbn2(self.lconv2(l)))
        r = F.relu(self.rbn1(self.rconv1(r)))
        r = self.rbn2(self.rconv2(r))
        r = F.relu(self.rbn3(self.rconv3(r)))
        h = F.concat((l, r), axis=1)
        return h


class Block(chainer.ChainList):

    def __init__(self, num_layers, in_channels, out_channels):
        super(Block, self).__init__()
        self.add_link(DownSampleUnit(in_channels, out_channels))
        for i in range(num_layers - 1):
            self.add_link(BasicUnit(out_channels))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


class ShuffleNetV2(chainer.Chain):

    insize = 224

    def __init__(self, scale_factor=1):
        super(ShuffleNetV2, self).__init__()

        initialW = initializers.HeNormal()
        out_channel_map = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
        assert scale_factor in out_channel_map, \
            'Unknown scale_factor: %f' % scale_factor
        out_channels = out_channel_map[scale_factor]

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 24, 3, stride=2, pad=1,
                                         initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(24)
            self.stage2 = Block(4, 24, out_channels[0])
            self.stage3 = Block(8, out_channels[0], out_channels[1])
            self.stage4 = Block(4, out_channels[1], out_channels[2])
            self.conv5 = L.Convolution2D(out_channels[2], out_channels[3], 1,
                                         initialW=initialW, nobias=True)
            self.bn5 = L.BatchNormalization(out_channels[3])
            self.fc = L.Linear(out_channels[3], 1000)

    def forward(self, x, t=None):
        h = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if t is None:
            return h

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


def main():
    import numpy as np
    import onnx_chainer

    #
    # model = BasicUnit(24)
    # print(model(np.random.rand(1, 24, 56, 56)).shape)
    # model = DownSampleUnit(24, 12)
    # print(model(np.random.rand(1, 24, 56, 56)).shape)
    #
    # model = ShuffleNetV2(1.0)
    # print(model(np.random.rand(1, 3, 224, 224).astype(np.float32), np.array([3])))
    model = ShuffleNetV2(2.0)
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    onnx_chainer.export_testcase(model, [x], 'shufflenet_v2_x2.0')


if __name__ == '__main__':
    main()
