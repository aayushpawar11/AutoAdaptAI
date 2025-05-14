import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    count = 0
    frame_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_id += 1
        count += 1

    cap.release()
    print(f"Extracted {frame_id} frames.")