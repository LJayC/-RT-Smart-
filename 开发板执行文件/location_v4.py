import gc
import os

import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from libs.PipeLine import ScopedTiming
from libs.Utils import *
from media.display import *
from media.media import *
from media.sensor import *

display_mode = "lcd"
if display_mode == "lcd":
    DISPLAY_WIDTH = ALIGN_UP(960, 16)
    DISPLAY_HEIGHT = 536
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = ALIGN_UP(640, 16)
OUT_RGB888P_HEIGH = 360

root_path = "/sdcard/mp_deployment_source/"
config_path = root_path + "deploy_config.json"
deploy_conf = {}
debug_mode = 1

# 定义正确的货物层位置
# 格式: {类别: 期望的层号}
# 层号: 0=最上层, 1=第二层, 2=第三层, 3=最下层
CORRECT_LAYERS = {
    "colo": 0,    # 最上层
    "juice": 0,   # 最上层
    "water": 0,   # 最上层
    "cookie": 1,  # 第二层
    "noodle": 2,  # 第三层
    "tobacco": 3  # 最下层
}

# 垂直位置差阈值 (像素)
LAYER_THRESHOLD = 50

def two_side_pad_param(input_size, output_size):
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    ratio = min(ratio_w, ratio_h)
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw - 0.1))
    return top, bottom, left, right, ratio

def read_deploy_config(config_path):
    with open(config_path, "r") as json_file:
        try:
            config = ujson.load(json_file)
        except ValueError as e:
            print("JSON parsing error:", e)
    return config

def check_position(det_boxes, labels):
    """检查货物层位置是否正确"""
    if not det_boxes:
        return True  # 没有检测到货物不算位置错误

    # 提取每个检测到的货物信息
    items = []
    for box in det_boxes:
        class_id = box[0]
        class_name = labels[class_id]

        # 转换为显示坐标
        x1_display = int(box[2] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
        y1_display = int(box[3] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)
        x2_display = int(box[4] * DISPLAY_WIDTH / OUT_RGB888P_WIDTH)
        y2_display = int(box[5] * DISPLAY_HEIGHT / OUT_RGB888P_HEIGH)

        center_y = (y1_display + y2_display) // 2

        items.append({
            "class_name": class_name,
            "center_y": center_y,
            "confidence": box[1]
        })

    # 按垂直位置排序 (从上到下)
    items.sort(key=lambda item: item["center_y"])

    # 找出每一层的货物
    layers = {}
    for item in items:
        # 根据垂直位置分组到不同层
        layer_found = False
        for layer_y, layer_items in layers.items():
            # 如果与现有层的垂直位置接近，归入该层
            if abs(item["center_y"] - layer_y) < LAYER_THRESHOLD:
                layer_items.append(item)
                layer_found = True
                break

        if not layer_found:
            # 创建新层
            layers[item["center_y"]] = [item]

    # 为每层分配层号 (0=最上层, 3=最下层)
    sorted_layers = sorted(layers.keys())
    layer_mapping = {y: idx for idx, y in enumerate(sorted_layers)}

    # 检查每个货物是否在正确的层
    for layer_y, items_in_layer in layers.items():
        layer_idx = layer_mapping[layer_y]

        for item in items_in_layer:
            class_name = item["class_name"]

            # 检查该货物是否应在当前层
            if class_name not in CORRECT_LAYERS:
                continue  # 未知类别，跳过

            expected_layer = CORRECT_LAYERS[class_name]

            # 检查层位置是否正确
            if expected_layer != layer_idx:
                return False

    return True

def detection():
    print("det_infer start")
    deploy_conf = read_deploy_config(config_path)
    kmodel_name = deploy_conf["kmodel_path"]
    labels = deploy_conf["categories"]
    confidence_threshold = deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    img_size = deploy_conf["img_size"]
    num_classes = deploy_conf["num_classes"]
    color_four = get_colors(num_classes)
    nms_option = deploy_conf["nms_option"]
    model_type = deploy_conf["model_type"]
    if model_type == "AnchorBaseDet":
        anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]
    kmodel_frame_size = img_size
    frame_size = [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH]
    strides = [8, 16, 32]

    top, bottom, left, right, ratio = two_side_pad_param(frame_size, kmodel_frame_size)

    kpu = nn.kpu()
    kpu.load_kmodel(root_path + kmodel_name)

    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build(
        [1, 3, OUT_RGB888P_HEIGH, OUT_RGB888P_WIDTH], [1, 3, kmodel_frame_size[1], kmodel_frame_size[0]]
    )

    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    sensor.set_framesize(width=OUT_RGB888P_WIDTH, height=OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

    sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)

    if display_mode == "lcd":
        Display.init(Display.NT35516, width=960, height=540, to_ide=True)
    else:
        Display.init(Display.LT9611, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)

    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    MediaManager.init()
    sensor.run()

    rgb888p_img = None
    ai2d_input_tensor = None
    data = np.ones((1, 3, kmodel_frame_size[1], kmodel_frame_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)

    # 物体统计变量
    class_counts = {}
    STATS_X = 10
    STATS_Y = 10
    LINE_HEIGHT = 30

    # 位置错误标志
    location_error = False

    # 错误计数和阈值
    error_count = 0
    ERROR_THRESHOLD = 5  # 连续5帧错误才显示错误

    while True:
        with ScopedTiming("total", debug_mode > 0):
            rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if rgb888p_img.format() == image.RGBP888:
                ai2d_input = rgb888p_img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()

                results = []
                for i in range(kpu.outputs_size()):
                    out_data = kpu.get_output_tensor(i)
                    result = out_data.to_numpy()
                    result = result.reshape((result.shape[0] * result.shape[1] * result.shape[2] * result.shape[3]))
                    del out_data
                    results.append(result)

                det_boxes = aicube.anchorbasedet_post_process(
                    results[0],
                    results[1],
                    results[2],
                    kmodel_frame_size,
                    frame_size,
                    strides,
                    num_classes,
                    confidence_threshold,
                    nms_threshold,
                    anchors,
                    nms_option,
                )

                osd_img.clear()
                frame_counts = {}  # 当前帧计数

                if det_boxes:
                    # 检查位置是否正确
                    current_position_correct = check_position(det_boxes, labels)

                    # 更新错误计数
                    if not current_position_correct:
                        error_count = min(error_count + 1, ERROR_THRESHOLD)
                    else:
                        error_count = max(error_count - 1, 0)

                    location_error = (error_count >= ERROR_THRESHOLD)

                    # 绘制检测框
                    for det_boxe in det_boxes:
                        x1, y1, x2, y2 = det_boxe[2], det_boxe[3], det_boxe[4], det_boxe[5]
                        x = int(x1 * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        y = int(y1 * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
                        w = int((x2 - x1) * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        h = int((y2 - y1) * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
                        osd_img.draw_rectangle(x, y, w, h, color=color_four[det_boxe[0]][1:])
                        text = labels[det_boxe[0]] + " " + str(round(det_boxe[1], 2))
                        osd_img.draw_string_advanced(x, y - 40, 32, text, color=color_four[det_boxe[0]][1:])

                        # 更新计数
                        class_id = det_boxe[0]
                        class_name = labels[class_id]
                        frame_counts[class_name] = frame_counts.get(class_name, 0) + 1
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # 绘制统计信息
                    current_y = STATS_Y

                    # 当前帧检测结果
                    osd_img.draw_string(STATS_X, current_y, "Current Frame:", scale=2, color=(0, 255, 0))
                    current_y += LINE_HEIGHT

                    for class_name, count in frame_counts.items():
                        text = f"{class_name}: {count}"
                        osd_img.draw_string(STATS_X + 20, current_y, text, scale=1.5, color=(0, 255, 0))
                        current_y += LINE_HEIGHT

                    # 全局统计结果
                    osd_img.draw_string(STATS_X, current_y, "Total Detection:", scale=2, color=(255, 255, 0))
                    current_y += LINE_HEIGHT

                    for class_name, total in class_counts.items():
                        text = f"{class_name}: {total}"
                        osd_img.draw_string(STATS_X + 20, current_y, text, scale=1.5, color=(255, 255, 0))
                        current_y += LINE_HEIGHT

                    # 检测总数
                    total_text = f"Total Objects: {len(det_boxes)}"
                    osd_img.draw_string(STATS_X, current_y, total_text, scale=2, color=(255, 0, 0))
                else:
                    # 没有检测到物体
                    location_error = False
                    error_count = 0
                    osd_img.draw_string(STATS_X, STATS_Y, "No objects detected", scale=2, color=(255, 0, 0))

                # 如果位置错误，显示错误信息
                if location_error:
                    # 在屏幕中央显示大红色错误提示
                    error_text = "Location error!!!"
                    text_width = len(error_text) * 40  # 估算文本宽度
                    text_x = max(0, (DISPLAY_WIDTH - text_width) // 2)
                    text_y = max(0, (DISPLAY_HEIGHT - 60) // 2)
                    osd_img.draw_rectangle(text_x - 20, text_y - 20, text_width + 40, 60, color=(255, 0, 0), fill=True)
                    osd_img.draw_string(text_x, text_y, error_text, scale=4, color=(255, 255, 255))

                Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
                gc.collect()
            rgb888p_img = None

    del ai2d_input_tensor
    del ai2d_output_tensor
    sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    gc.collect()
    time.sleep(1)
    nn.shrink_memory_pool()
    print("det_infer end")
    return 0

if __name__ == "__main__":
    detection()
