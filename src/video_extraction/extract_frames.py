import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=1):
    print(f"🔍 Loading video: {video_path}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ FPS: {fps}")
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
            print(f"🖼️ Saved: {frame_path}")
            frame_id += 1
        count += 1

    cap.release()
    print(f"✅ Done. Extracted {frame_id} frames.")

if __name__ == "__main__":
    extract_frames("sample_video.mp4", "data/frames")
