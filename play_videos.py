import cv2
import os

video_folder = "cifar10_videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]

for video_file in video_files:
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Playback', frame)
        
        # Increase the waitKey delay to make sure it registers the 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()  # Close the video window before loading the next one

cv2.destroyAllWindows()
