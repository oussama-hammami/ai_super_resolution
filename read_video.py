import cv2
import config, time

def read_video_from_file(video_path:str)->list:
    captured =  cv2.VideoCapture(video_path)
    print(captured, flush = True)
    if not captured.isOpened:
        raise Exception("Video read unsuccessful!") 
    return captured

def calculate_fps(frame_upscaled, prev_time):
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(frame_upscaled, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return prev_time


    