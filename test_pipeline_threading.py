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


stop_event = threading.Event()

captured = read_video_from_file(config.video_path)

# Model preparation and warm up
sr_model = prepare_sr_model()
sr_model.predict(torch.zeros(1,3,480,540).to("cuda"))

def load_and_prepare_video_frames(q1, captured):
    while not stop_event.is_set():
        ret, lr_image = captured.read()
        if not ret:
            q1.put(None)
            break
        lr_image = lr_image[:, :, ::-1].copy()  # BGR -> RGB
        lr_image = np.transpose(lr_image, (2, 0, 1))
        lr_image = torch.from_numpy(lr_image)
        lr_image = gaussian_blur(lr_image, sigma=0.3)
        lr_image = rgb2ycbcr(lr_image)
        lr_image = norm01(lr_image)
        lr_image = torch.unsqueeze(lr_image, dim=0).to(config.device)
        q1.put(lr_image)
    if stop_event.is_set():
        q1.put(None)

def upscale_frame(q1, q2, model):
    while True:
        lr_image = q1.get()
        if lr_image is None:
            q2.put(None)
            q1.task_done()
            break
        with torch.no_grad():
            sr_image = model.predict(lr_image)[0]
        sr_image = denorm01(sr_image)
        sr_image = sr_image.type(torch.uint8)
        sr_image = ycbcr2rgb(sr_image)  # YCbCr -> RGB
        sr_image = sr_image.cpu().detach().numpy()
        sr_image = np.transpose(sr_image, (1, 2, 0))
        sr_image = sr_image[:, :, ::-1].copy()  # RGB -> BGR for cv2
        q2.put(sr_image)
        q1.task_done()

def display_frames(q2):
    prev_time = time.time()
    while True:
        upscaled_frame = q2.get()
        if upscaled_frame is None:
            q2.task_done()
            break
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(upscaled_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video_frame", upscaled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            q2.task_done()
            break
        q2.task_done()

lr_to_sr = Queue(maxsize=10)
sr_to_display = Queue(maxsize=10)

frame_preparation_thread = threading.Thread(target = load_and_prepare_video_frames,
                                            args = (lr_to_sr, captured)) 
run_inference_thread = threading.Thread(target = upscale_frame,
                                        args = (lr_to_sr,sr_to_display,sr_model))
display_frames_thread = threading.Thread(target = display_frames,
                                         args = (sr_to_display,))
frame_preparation_thread.start()
run_inference_thread.start()
display_frames_thread.start()

frame_preparation_thread.join()

lr_to_sr.join()
sr_to_display.join()

run_inference_thread.join()
display_frames_thread.join()
