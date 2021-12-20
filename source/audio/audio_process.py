import numpy as np
import librosa
import time
import pyaudio
import wave
import os
import threading

import paddle
from paddle import nn
from source.audio.cnn_model import CNN, SpeechCommandModel

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
paddle.set_device('gpu')

#模拟消费数据
def Consumer(q):
    while True:
        while True:
            print("consume: ", 1)
            res = q.get()
            print("received data: ", res)
        time.sleep(1)


class stateManager(object):
    def __init__(self):
        '''
        左右方向不同时至少两次连续才能确认
        action_list 状态索引顺序: '左', '右', '下', '停', '跑', '跳', '打', '其它'

        [1, 0, 0, 0, 0, 0, 0] action, jump, left, right, up, down, return
        '''
        self.reset()

    #重置counter
    def reset_counter(self):
        self.counter = {
            "左": 0,
            "右": 0,
            "下": 0,
            "跳": 0,
        }

    # 停止
    def reset(self):
        self.reset_counter()
        self.current_state = [0, 0, 0, 0, 0, 0, 0]

    #重置一次性操作
    def reset_once(self):
        for idx in [0, 4, 6]: #action, up, return
            self.current_state[idx] = 0

    # 向左
    def reset_for_left(self):
        left_times = self.counter["左"]
        self.reset()
        self.counter["左"] = left_times
        self.current_state[2] = 1 #left

    # 向右
    def reset_for_right(self):
        left_times = self.counter["右"]
        self.reset()
        self.counter["右"] = left_times
        self.current_state[3] = 1 #right

    # 下蹲
    def reset_for_down(self):
        down_times = self.counter["下"]
        self.reset()
        self.counter["下"] = down_times
        self.current_state[5] = 1 #down

    # 跑 组合键, 跑之后必须配合停止, 跑时没有下蹲
    def ctrl_for_run(self):
        self.current_state[0] = 1 #action
        for idx in [4, 5]:
            self.current_state[idx] = 0

    # 跳 有方向时终止, 否则会一直长按(大跳), 如跳+左 就是小跳
    def ctrl_for_jump(self):
        self.current_state[1] = 1 #jump
        jump_times = self.counter["跳"]
        self.reset_counter()
        self.counter["跳"] = jump_times
        for idx in [2, 3, 5]: #跳 下
            self.current_state[idx] = 0

    #根据状态获取动作
    def get_action(self, state_idx):
        if state_idx == 0: #左
            self.counter["左"] += 1
            if self.counter["左"] > 1:
                self.reset_for_left()
        elif state_idx == 1:
            self.counter["右"] += 1
            if self.counter["右"] > 1:
                self.reset_for_right()
        elif state_idx == 2:
            self.counter["下"] += 1
            if self.counter["下"] > 1:
                self.reset_for_down()
        elif state_idx == 3: #停
            self.reset()
        elif state_idx == 4: #跑
            self.ctrl_for_run()
        elif state_idx == 5: #跳
            self.counter["跳"] += 1
            if self.counter["跳"] > 1:
                self.ctrl_for_jump()
        elif state_idx == 6: #打, 只是一次性动作
            self.current_state[0] = 1

        res = self.current_state
        self.reset_once()
        print("send current: ", res)
        return res

    #持续上次动作
    def get_last_action(self):
        print("send last: ", self.current_state)
        return self.current_state


class inferThread (threading.Thread):
    def __init__(self, w_q, a_q):
        self.action_que = a_q  # 面向pygame发送数据
        self.wav_que = w_q
        self.model = SpeechCommandModel(num_classes=8)  # 初始化模型
        self.labels = ['左', '右', '下', '停', '跑', '跳', '打', '其它']
        state_dict = paddle.load("models/final.pdparams")
        self.model.set_state_dict(state_dict)
        self.model.eval()
        self.state_manager = stateManager()

        threading.Thread.__init__(self)

    #数据处理
    def preprocess(self, audio_bytes, sampling_rate=16000):
        t1 = time.time()
        # wav, sr = librosa.load(file_path, sr=sampling_rate)
        # print(wav.shape)
        # exit()
        n_bytes = 2
        scale = 1.0 / float(1 << ((8 * n_bytes) - 1))
        fmt = "<i{:d}".format(n_bytes)
        wav = scale * np.frombuffer(audio_bytes, fmt).astype(np.float32)
        print("max: ", np.max(wav))

        intervals = librosa.effects.split(wav, top_db=20)
        wav_output = []
        for sliced in intervals:
            wav_output.extend(wav[sliced[0]:sliced[1]])

        wav_len = int(sampling_rate / 2)
        if len(wav_output) > wav_len:
            wav_output = np.array(wav_output)[:wav_len]
        else:
            wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
            wav_output = np.array(wav_output)
        # print("consume time: ", time.time() - t1)
        ps = librosa.feature.melspectrogram(y=wav_output, sr=sampling_rate, hop_length=128, n_fft=1024, n_mels=80,
                                            power=1.0, fmin=40.0, fmax=sampling_rate / 2).astype(np.float32)

        ps = np.expand_dims(ps, axis=-1).transpose((1, 0, 2))  # 126*80*1
        # print("convert time: ", time.time() - t1)
        return paddle.to_tensor(np.expand_dims(ps, axis=0))


    def run(self):
        while True:

            audio_bytes = self.wav_que.get()
            t1 = time.time()
            input_data = self.preprocess(audio_bytes)
            if input_data is None:
                self.action_que.put(self.state_manager.get_last_action(), True, 2)
                continue

            output = self.model(input_data)
            output = output.numpy()
            # print (output)
            max_idx = np.argmax(output)
            # print (max_idx)

            label = self.labels[max_idx]
            print("probe", max_idx, label, output[0][max_idx])
            print("time consuming...", time.time() - t1, self.action_que.qsize(), self.wav_que.qsize())
            if output[0][max_idx] < 0.4: #置信度太低
                self.action_que.put(self.state_manager.get_last_action(), True, 2)
                continue

            # 解析output并转为action数据
            self.action_que.put(self.state_manager.get_action(max_idx), True, 2)

            continue

class ReadAudio(threading.Thread):
    CHUNK = 1204
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 0.5 #500ms来一次

    def __init__(self,q):
        self.wav_que=q
        self.record_interval = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
        threading.Thread.__init__(self)

    def run(self):
        print("* recording")
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        # last_time = 0
        while True:
            raw_data = []
            for i in range(0, self.record_interval):
                data = stream.read(self.CHUNK)
                raw_data.append(data)
            self.wav_que.put(b''.join(raw_data))
            # print("record_time", time.time() - last_time)
            # last_time = time.time()
            # print("wav_queue size: ", self.wav_que.qsize())
            # print(1)

        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    wav_q = Queue(100)
    act_q = Queue(100)

    infer_thread = inferThread(wav_q, act_q)
    infer_thread.start()

    read_audio = ReadAudio(wav_q)
    read_audio.start()

    consumer = Consumer(act_q)
    consumer.start()

    time.sleep(5)

    print("主进程")



