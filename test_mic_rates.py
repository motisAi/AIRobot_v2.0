"""Test opening camera mic with PyAudio at different rates."""
import pyaudio

p = pyaudio.PyAudio()
device = 0

for rate in [44100, 48000, 16000, 22050, 8000]:
    for channels in [1, 2]:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=512,
                input_device_index=device,
            )
            data = stream.read(512, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()
            print(f"OK: device={device} rate={rate} channels={channels} read={len(data)} bytes")
        except Exception as e:
            print(f"FAIL: device={device} rate={rate} channels={channels} -> {e}")

p.terminate()
