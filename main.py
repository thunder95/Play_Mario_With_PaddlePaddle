import pygame as pg
from source.main import main
from source.audio.audio_process import inferThread, ReadAudio
import signal
from multiprocessing import Queue
import sys
from source.video.video_process import Video_Process

if __name__=='__main__':
    is_video = True

    if len(sys.argv) > 1 and sys.argv[1] == "audio":
        is_video = False

    que = Queue(100)

    if is_video:
        #姿态
        vp = Video_Process(que)
        # 启动视频解析主程序
        vp.start()
    else:
        #语音
        wav_q = Queue(100)
        infer_thread = inferThread(wav_q, que)
        infer_thread.start()

        read_audio = ReadAudio(wav_q)
        read_audio.start()

    main(que)
    pg.quit





