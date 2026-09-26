import cv2
import time
from FSRCNN_Pytorch.utils.common import *
import config
from read_video import read_video_from_file, calculate_fps
from prep_sr_model import prepare_sr_model, run_inferance
from queue import Queue
import torch
import numpy as np
import threading

BATCH_SIZE = 2

stop_event = threading.Event()

captured = read_video_from_file(config.video_path)

# Model preparation and warm up
sr_model = prepare_sr_model()
sr_model.predict(torch.zeros(1, 3, 480, 540).to("cuda"))


# -----------------------------------------------------------
# Thread 1: read raw frames into raw_queue
# -----------------------------------------------------------
def read_frames(raw_queue, captured):
    while not stop_event.is_set():
        ret, frame = captured.read()
        if not ret:
            raw_queue.put(None)
            break
        raw_queue.put(frame)
    if stop_event.is_set():
        raw_queue.put(None)


# -----------------------------------------------------------
# Thread 2: pre-process raw frames → lr tensors
# -----------------------------------------------------------
def preprocess_frames(raw_queue, q1):
    while True:
        frame = raw_queue.get()
        if frame is None:
            q1.put(None)
            raw_queue.task_done()
            break
        lr_image = frame[:, :, ::-1].copy()          # BGR -> RGB
        lr_image = np.transpose(lr_image, (2, 0, 1))
        lr_image = torch.from_numpy(lr_image)
        lr_image = gaussian_blur(lr_image, sigma=0.3)
        lr_image = rgb2ycbcr(lr_image)
        lr_image = norm01(lr_image)
        lr_image = torch.unsqueeze(lr_image, dim=0).to(config.device)
        q1.put(lr_image)
        raw_queue.task_done()


# -----------------------------------------------------------
# Thread 3: batch inference + GPU post-process, put HWC BGR GPU tensors to q2
# -----------------------------------------------------------
def infer_batch(batch, q2, model):
    stacked = torch.cat(batch, dim=0)
    with torch.no_grad():
        sr_batch = model.predict(stacked)
        for i in range(sr_batch.shape[0]):
            sr_image = denorm01(sr_batch[i])
            sr_image = sr_image.type(torch.uint8)
            sr_image = ycbcr2rgb(sr_image)                         # YCbCr -> RGB
            # RGB -> BGR and CHW -> HWC on the GPU, so the download is already cv2 layout
            sr_image = sr_image[[2, 1, 0]].permute(1, 2, 0).contiguous()
            q2.put(sr_image)
    batch.clear()


def upscale_frames(q1, q2, model):
    batch = []
    while True:
        lr_image = q1.get()

        if lr_image is None:
            # flush remaining batch
            if batch:
                infer_batch(batch, q2, model)
            q2.put(None)
            q1.task_done()
            break

        batch.append(lr_image)
        q1.task_done()

        if len(batch) >= BATCH_SIZE:
            infer_batch(batch, q2, model)


# -----------------------------------------------------------
# Thread 4: download + display
# -----------------------------------------------------------
def display_frames(q2):
    prev_time = time.time()
    host_buffer = None
    while True:
        sr_image = q2.get()
        if sr_image is None:
            q2.task_done()
            break

        # single pinned host buffer, allocated once and reused every frame
        if host_buffer is None:
            host_buffer = torch.empty(sr_image.shape, dtype=torch.uint8, pin_memory=True)
        host_buffer.copy_(sr_image)
        sr_image = host_buffer.numpy()

        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(sr_image, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video_frame", sr_image)
        if cv2.pollKey() & 0xFF == ord('q'):
            stop_event.set()
            q2.task_done()
            break
        q2.task_done()


raw_frames   = Queue(maxsize=30)
lr_to_sr     = Queue(maxsize=30)
sr_to_display = Queue(maxsize=30)

read_thread        = threading.Thread(target=read_frames,       args=(raw_frames, captured))
preprocess_thread  = threading.Thread(target=preprocess_frames, args=(raw_frames, lr_to_sr))
inference_thread   = threading.Thread(target=upscale_frames,    args=(lr_to_sr, sr_to_display, sr_model))
display_thread     = threading.Thread(target=display_frames,    args=(sr_to_display,))

read_thread.start()
preprocess_thread.start()
inference_thread.start()
display_thread.start()

read_thread.join()
raw_frames.join()
lr_to_sr.join()
sr_to_display.join()

preprocess_thread.join()
inference_thread.join()
display_thread.join()

captured.release()
cv2.destroyAllWindows()
