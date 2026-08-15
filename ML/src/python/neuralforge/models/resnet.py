from ..nn.convolution import ResNetBlock


def ResNet18(num_classes=1000, in_channels=3):
    from ..nn.convolution import ResNet

    return ResNet(ResNetBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes=1000, in_channels=3):
    from ..nn.convolution import ResNet

    return ResNet(ResNetBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes=1000, in_channels=3):
    from ..nn.convolution import ResNet
    from ..nn.layers import BottleneckBlock

    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes, in_channels)
