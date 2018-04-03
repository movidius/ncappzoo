import tensorflow as tf
import tensorflow.contrib as tc
from .common import SCOPE


#
# class Block(nn.Module):
#     '''expand + depthwise + pointwise'''
#
#     def __init__(self, in_planes, out_planes, expansion, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#
#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)
#
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride == 1 else out
#         return out

def batch_norm(x, name, training):
    return tf.layers.batch_normalization(x, name=name, training=training, momentum=0.95)


def block(x: tf.Tensor, out_planes, expansion, stride, is_training):
    in_planes: tf.Dimension = x.shape[3]
    planes = expansion * in_planes.value

    shortcut = x

    # conv1
    x = tf.layers.conv2d(x, planes, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1')
    x = batch_norm(x, name='conv1_bn', training=is_training)
    x = tf.nn.relu(x)

    # conv2 (depthwise)
    x = tf.layers.separable_conv2d(x, out_planes, kernel_size=3, strides=stride, padding='same', use_bias=False,
                                   name='conv2')
    x = batch_norm(x, name='conv2_bn', training=is_training)
    x = tf.nn.relu(x)

    # shortcut
    if stride == 1 and in_planes != out_planes:
        shortcut = tf.layers.conv2d(shortcut, out_planes, kernel_size=1, strides=1, padding='same', use_bias=False,
                                    name='shortcut')
        shortcut = batch_norm(shortcut, training=is_training, name='shortcut_bn')
    x = tf.add(x, shortcut if stride == 1 else x)
    return x


#
#
# class MobileNetV2(nn.Module):
#     # (expansion, out_planes, num_blocks, stride)
#     cfg = [(1, 16, 1, 1),
#            (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
#            (6, 32, 3, 2),
#            (6, 64, 4, 2),
#            (6, 96, 3, 1),
#            (6, 160, 3, 2),
#            (6, 320, 1, 1)]
#
#     def __init__(self, num_classes=10):
#         super(MobileNetV2, self).__init__()
#         # NOTE: change conv1 stride 2 -> 1 for CIFAR10
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(1280)
#         self.linear = nn.Linear(1280, num_classes)
#
#     def _make_layers(self, in_planes):
#         layers = []
#         for expansion, out_planes, num_blocks, stride in self.cfg:
#             strides = [stride] + [1] * (num_blocks - 1)
#             for stride in strides:
#                 layers.append(Block(in_planes, out_planes, expansion, stride))
#                 in_planes = out_planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


def mobilenet_v2_cifar10(is_training, images, num_classes):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(0, 32, 0, 1),  # conv1, only out_planes and stride

           (1, 16, 1, 1),  # block layers
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1),

           (0, 1280, 0, 0)]  # last conv layer, only out_planes
    return mobilenet_v2_base(is_training, images, num_classes, cfg)


def mobilenet_v2_base(is_training, images, num_classes, cfg):
    with tf.variable_scope(SCOPE):
        # conv1
        x = tf.layers.conv2d(images, cfg[0][1], kernel_size=3, strides=cfg[0][3], padding='same', use_bias=False,
                             name='conv1')
        x = batch_norm(x, name='conv1_bn', training=is_training)
        x = tf.nn.relu(x)

        # layers
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(cfg[1:-1]):
            strides = [stride] + [1] * (num_blocks - 1)
            for j, _stride in enumerate(strides):
                with tf.variable_scope(f'block{i}_{j}'):
                    x = block(x, out_planes, expansion, _stride, is_training)

        # bottleneck
        x = tf.layers.conv2d(images, cfg[-1][1], kernel_size=1, strides=1, padding='same', use_bias=False,
                             name='bottleneck')
        x = batch_norm(x, name='bottleneck_bn', training=is_training)
        x = tf.nn.relu(x)

        # global pool
        k = (x.shape[1], x.shape[2])
        x = tf.layers.average_pooling2d(x, pool_size=k, strides=k, name='pool')

        # logits
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, num_classes, name='logits')

        # proba
        p = tf.nn.softmax(x, name='proba')
        return x, p
