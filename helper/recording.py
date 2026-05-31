import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# ----------------------------
# Configuration
# ----------------------------
output_dir = "../data/test"
person_id = input("Enter person ID (e.g., person01): ")
num_sequences = int(input("How many sequences for this person? "))

person_dir = os.path.join(output_dir, person_id)
os.makedirs(person_dir, exist_ok=True)

# ----------------------------
# RealSense setup
# ----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Warm up
for _ in range(30):
    pipeline.wait_for_frames()

# ----------------------------
# Recording loop
# ----------------------------
try:
    for seq_num in range(1, num_sequences + 1):
        print(f"\nReady to record sequence {seq_num}.")
        print("Press 's' to start/stop recording. Press 'q' to quit.")

        recording = False
        depth_frames = []
        timestamps = []

        rgb_filename = os.path.join(person_dir, f"seq{seq_num:02d}_rgb.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_rgb = cv2.VideoWriter(rgb_filename, fourcc, 30, (640, 480))

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Normalize depth for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Side-by-side live preview
            display_image = np.hstack((color_image, depth_colormap))
            cv2.putText(display_image, "Press 's' to start/stop, 'q' to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if recording:
                out_rgb.write(color_image)
                depth_frames.append(depth_image.copy())
                timestamps.append(time.time())
                cv2.putText(display_image, "Recording...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("RealSense Recording", display_image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('s'):
                recording = not recording
                if not recording and depth_frames:
                    depth_filename = os.path.join(person_dir, f"seq{seq_num:02d}_depth.npy")
                    timestamps_filename = os.path.join(person_dir, f"seq{seq_num:02d}_timestamps.npy")
                    np.save(depth_filename, np.stack(depth_frames))
                    np.save(timestamps_filename, np.array(timestamps))
                    print(f"Sequence {seq_num} saved!")
                    break

            if key & 0xFF == ord('q'):
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

finally:
    pipeline.stop()
    out_rgb.release()
    cv2.destroyAllWindows()
