import cv2
import time
import config
from read_video import read_video_from_file, calculate_fps
from prep_sr_model import prepare_sr_model, run_inferance

captured = read_video_from_file(config.video_path)
sr_model = prepare_sr_model()
prev_time = time.time()
while True:
    ret,frame = captured.read()
    if not ret:
        break
    frame_upscaled  = run_inferance(model=sr_model, lr_image=frame)

    prev_time = calculate_fps(frame_upscaled=frame_upscaled, prev_time=prev_time)

    cv2.imshow("video_frame", frame_upscaled)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close any OpenCV windows
captured.release()
cv2.destroyAllWindows()