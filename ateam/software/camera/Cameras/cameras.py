import cv2
import numpy as np

def high_perf_3cam():
    # 1. Setup Camera Indices
    cam_indices = [0, 1, 2]
    caps = []
    
    for idx in cam_indices:
        cap = cv2.VideoCapture(idx)
        
        # FIX LAG: Force MJPG compression at the hardware level
        # This is the most important step for 3 webcams on one PC
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set Resolution (1080p for clear video)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Set FPS to 30 to ensure smooth movement
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Buffer size: set to 1 to ensure you see the 'current' frame, not a delayed one
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        caps.append(cap)

    print("High-Performance View Started. Press 'q' to quit.")

    # Create a named window and force it to be fullscreen or large
    cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frames.append(frame)

        # Combine them horizontally
        combined = np.hstack(frames)

        # MAKE IT BIGGER: 
        # Instead of shrinking to 0.3, we use a larger scale (e.g., 0.6 or 0.8)
        # 0.6 will make a 3456 pixel wide preview. 
        # If it's still too small, change 0.5 to a higher number.
        scale = 0.5 
        h, w = combined.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        
        # Use INTER_LINEAR for faster, cleaner resizing
        large_preview = cv2.resize(combined, new_size, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Robot View", large_preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    high_perf_3cam()