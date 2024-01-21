import sensor
import time
import image
import math
import gc
from pyb import UART


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)
sensor.skip_frames(time=2000)
clock = time.clock()

aruco_dict_4x4 = {(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0): 123, (0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0): 213, (0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0): 214, (0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0): 234, (0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 0, 0, 255, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0): 341, (0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0): 342, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): 412, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 255, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0): 413}

def get_aruco_code(aruco_img_):
    code_variants = list()
    for i in range(4):
        aruco_arr = tuple(b for b in aruco_img_.bytearray())
        code_variants.append(aruco_arr)
        aruco_img_ = aruco_img_.replace(vflip=True, hmirror=False, transpose=True)
    aruco_img_ = aruco_img_.replace(vflip=True, hmirror=False, transpose=False)
    for i in range(4):
        aruco_arr = tuple(b for b in aruco_img_.bytearray())
        code_variants.append(aruco_arr)
        aruco_img_ = aruco_img_.replace(vflip=True, hmirror=False, transpose=True)
    aruco_codes = list(aruco_dict_4x4.keys())
    hamming_dists = list()
    for i, code in enumerate(code_variants):
        for j, aruco_code in enumerate(aruco_codes):
            dist = sum([code[k] != aruco_code[k] for k in range(36)])
            hamming_dists.append((i, j, dist))
    hamming_dists.sort(key=lambda s: s[2])
    # print(code_variants)
    # print(hamming_dists)

    if hamming_dists[0][2] < 3:
        return aruco_dict_4x4.get(aruco_codes[hamming_dists[0][1]])
    else:
        return -1

def recognize_aruco(img):
    rects = img.find_rects(threshold=10000)
    code = None
    if rects:
        r_max = max(rects, key=lambda x: x.w() * x.h())
        norm_rect = img.copy()
        # print(tuple(reversed(r_max.corners())))
        r_norm_rect = norm_rect.rotation_corr(corners=tuple(reversed(r_max.corners())))
        min_side = min(r_norm_rect.width(), r_norm_rect.height())
        k = 0.7
        r_norm_rect = r_norm_rect.to_grayscale(
            x_scale=(min_side / r_norm_rect.width()) * k,
            y_scale=(min_side / r_norm_rect.height()) * k,
            hint=image.BILINEAR,
        )

        segment = round(round((min_side / 6)) * k)
        r_norm_rect = r_norm_rect.to_bitmap()
        pixels_counter = [[0 for _ in range(6)] for __ in range(6)]
        # print(r_norm_rect.width(), r_norm_rect.height())
        counter = 0
        for x in range(r_norm_rect.width()):
            for y in range(r_norm_rect.height()):
                pixels_counter[x // segment][y // segment] += r_norm_rect.get_pixel(
                    x, y
                )
                counter += r_norm_rect.get_pixel(x, y)
        # r_norm_rect = r_norm_rect.scale(x_scale=0.4,y_scale=0.4, hint=image.BILINEAR)

        max_pixel = 0
        min_pixel = pixels_counter[0][0]
        aruco_img = image.Image(6, 6, image.GRAYSCALE)

        for y, line in enumerate(pixels_counter):
            for x, v in enumerate(line):
                aruco_img.set_pixel(x, y, v)
            max_pixel = max(max_pixel, max(line))
            min_pixel = min(min_pixel, min(line))
        threshold = int((max_pixel + min_pixel) / 2)
        aruco_img = aruco_img.binary([(0, threshold)]).invert()

        code = get_aruco_code(aruco_img)
        img.draw_image(aruco_img, 0, 0)

        del pixels_counter, r_norm_rect, aruco_img
    del img, rects
    gc.collect()
    return code


uart = UART(3, 19200)
code_to_ev3code = {123: 11, 213: 13, 214: 17, 234: 19, 341: 23, 342: 29, 412: 31, 413: 37}
while True:
    if uart.any():
        b = uart.read()
        command = int.from_bytes(b, "big")
        if command == 7:
            img = sensor.snapshot()
            aruco_code = recognize_aruco(img)
            if aruco_code == -1 or aruco_code is None:
                ev3code = 44
            else:
                ev3code = code_to_ev3code[aruco_code]
            v = ev3code.to_bytes(1, 'big')
            uart.write(v)
            time.sleep_ms(50)

# 44 - error
# 7 - command get aruco id
