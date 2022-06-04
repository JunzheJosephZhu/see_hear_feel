import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('./data/test_recordings_0123/2022-01-23 15_33_20.205659/data.hdf5', 'r')

# cam_fixed_writer = cv2.VideoWriter(
#     "./test_recordings/2022-02-06 23:27:58.713705/cam_fixed.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (640, 480)
# )
# print(f"key: {f['cam_fixed_color'].shape}")
cnt = 0
for s in f['left_gelsight_flow'].iter_chunks():
    # print(s, f['cam_fixed_color'][s].shape)
    # plt.imshow(f['cam_fixed_color'][s])
    flow_data = f["left_gelsight_flow"][s]
    Ox = flow_data[0, :].reshape(-1)
    Oy = flow_data[1, :].reshape(-1)
    gx = flow_data[2, :].reshape(-1)
    gy = flow_data[3, :].reshape(-1)

    dx = gx - Ox
    dy = gy - Oy
    # plt.pause(.1)
    # plt.close()
    cnt += 1
    if cnt > 90 and cnt < 101:
        plt.figure(cnt)
        plt.quiver(Ox, Oy, dx, dy, angles='xy', scale=0.5, scale_units='xy')
    elif cnt == 101:
        break

plt.show()

f.close()