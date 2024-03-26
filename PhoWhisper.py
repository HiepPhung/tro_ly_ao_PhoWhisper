from transformers import pipeline
import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

# Thiết lập các tham số ghi âm
fs = 16000  # Tần số mẫu (samples per second)
duration = 5  # Thời lượng ghi âm (seconds)

# Đặt biến flag để kiểm tra khi nào thoát khỏi vòng lặp
running = True

while running:
    # Ghi âm từ microphone
    print("Bắt đầu ghi âm...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Chờ cho đến khi ghi âm hoàn thành

    # Lưu dữ liệu âm thanh vào một file WAV
    write('recorded_audio.wav', fs, recording)
    print("Ghi âm hoàn thành và lưu vào file 'recorded_audio.wav'")

    # Speech to text
    '''
    Model:
    vinai/PhoWhisper-tiny
    vinai/PhoWhisper-base
    vinai/PhoWhisper-small
    vinai/PhoWhisper-medium
    vinai/PhoWhisper-large
    '''
    transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device='cuda')
    output = transcriber("recorded_audio.wav")['text']
    print("Kết quả recorded: {}".format(output))

    # Xóa file ghi âm, giải phóng bộ nhớ
    os.remove("recorded_audio.wav")
    time.sleep(3)   # Đợi 3s rồi tiếp tục ghi

    # Thoát vòng lặp while
    if "tắt" in output:
        running = False