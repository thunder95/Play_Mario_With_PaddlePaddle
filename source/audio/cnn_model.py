from paddle import nn
import paddle
import paddle.nn.functional as F


class CNN(nn.Layer):
    def __init__(self, num_classes=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(
                in_channels=1,  # 输入为单层图像
                out_channels=16,  # 卷积成16层
                kernel_size=5,  # 卷积壳5x5
                stride=1,  # 步长，每次移动1步
                padding=2,  # 边缘层，给图像边缘增加像素值为0的框
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2D(kernel_size=2),  # 池化层，将图像长宽减少一半
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2D(2),
        )
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = paddle.reshape(x, shape=[x.shape[0], -1])
        output = self.out(x)
        return output


class SpeechCommandModel(nn.Layer):
    def __init__(self, num_classes=10):
        super(SpeechCommandModel, self).__init__()
        self.conv1 = nn.Conv2D(63, 10, (5, 1), padding="SAME")
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2D(10)

        self.conv2 = nn.Conv2D(10, 1, (5, 1), padding="SAME")
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2D(1)

        self.lstm1 = nn.LSTM(input_size=80,
                             hidden_size=64,
                             direction="bidirect")

        self.lstm2 = nn.LSTM(input_size=128,
                             hidden_size=64,
                             direction="bidirect")

        self.query = nn.Linear(128, 128)
        self.softmax = nn.Softmax(axis=-1)

        self.fc1 = nn.Linear(128, 64)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, num_classes)
        self.cls_softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        # print(x.shape)

        x = x.squeeze(axis=-1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x.squeeze(axis=1)

        q = self.query(x)
        attScores = paddle.matmul(q, x, transpose_y=True)
        attScores = self.softmax(attScores)
        attVector = paddle.matmul(attScores, x)
        # print(attVector.shape)

        output = self.fc1(attVector)
        output = self.fc1_relu(output)
        output = self.fc2(output)
        output = self.classifier(output)
        output = self.cls_softmax(output)
        # print(output)

        return output


class SpeechLstm(nn.Layer):
    def __init__(self, num_classes=10):
        super(SpeechLstm, self).__init__()
        self.lstm = nn.LSTM(12, 64, 3, time_major=False)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        rnn = paddle.nn.LSTM(16, 32, 2)
        x = paddle.randn((4, 23, 16))
        prev_h = paddle.randn((2, 4, 32))
        prev_c = paddle.randn((2, 4, 32))
        y, (h, c) = rnn(x, (prev_h, prev_c))
        """
        # print(x.shape)
        x = paddle.transpose(x, perm=[0, 2, 1])
        batch_size = x.shape[0]
        h0 = paddle.zeros(shape=[3, batch_size, 64])
        c0 = paddle.zeros(shape=[3, batch_size, 64])
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def save_model(net, path):
    paddle.save(net, path)


def load_model(path):
    net = paddle.load(path)
    return net


if __name__ == "__main__":
    model = CNN()
    print(model)
