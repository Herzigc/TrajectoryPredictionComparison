import numpy as np
import cv2
import os

root = '../data/final/test_filled'

for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith("depth.npy"):
            depth_data = np.load(os.path.abspath(os.path.join(root, file)))

            # Normalize depth values for visualization
            depth_min = np.min(depth_data)
            depth_max = np.max(depth_data)

            # Create a video writer
            height, width = depth_data.shape[1], depth_data.shape[2]
            fps = 30  # frames per second
            out = cv2.VideoWriter(os.path.dirname(os.path.abspath(os.path.join(root, file))) + '\\' + os.path.splitext((os.path.basename(file)))[0] + '.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=True)

            for frame in depth_data:
                normalized_frame = ((frame - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                color_frame = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)
                out.write(color_frame)  # now it works

            out.release()
            cv2.destroyAllWindows()
            print('Video saved as ' + os.path.dirname(os.path.abspath(os.path.join(root, file))) + '\\' + os.path.splitext((os.path.basename(file)))[0] + '.avi')