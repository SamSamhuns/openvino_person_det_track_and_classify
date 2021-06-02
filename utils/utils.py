import os
import argparse

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore


def parse_arguments(desc):
    # Parse Arguments
    parser = argparse.ArgumentParser(description=desc)
    # person detection models
    parser.add_argument('--pdet-model-xml',
                        # default='models/person-detection-retail-0013/person-detection-retail-0013.xml',  # 31 fps
                        # default='models/person-detection-0200/person-detection-0200.xml',                # 37 fps
                        default='models/person-detection-0201/person-detection-0201.xml',                # 30 fps
                        # default='models/person-detection-0202/person-detection-0202.xml',                # 23 FPS
                        help='XML File')
    parser.add_argument('--pdet-model-bin',
                        # default='models/person-detection-retail-0013/person-detection-retail-0013.bin',  # 31 fps
                        # default='models/person-detection-0200/person-detection-0200.bin',                # 37 fps
                        default='models/person-detection-0201/person-detection-0201.bin',                # 30 fps
                        # default='models/person-detection-0202/person-detection-0202.bin',                # 23 FPS
                        help='BIN File')
    # person re-identification models
    parser.add_argument('--preid-model-xml',
                        # default="models/person-reidentification-retail-0277-fp32/person-reidentification-retail-0277.xml",  # slowest
                        default="models/person-reidentification-retail-0286-fp32/person-reidentification-retail-0286.xml",  # balanced
                        # default="models/person-reidentification-retail-0288-fp32/person-reidentification-retail-0288.xml",  # fastest
                        help='XML File')
    parser.add_argument('--preid-model-bin',
                        # default="models/person-reidentification-retail-0277-fp32/person-reidentification-retail-0277.bin",  # slowest
                        default="models/person-reidentification-retail-0286-fp32/person-reidentification-retail-0286.bin",  # balanced
                        # default="models/person-reidentification-retail-0288-fp32/person-reidentification-retail-0288.bin",  # fastest
                        help='BIN File')
    # general arguments
    parser.add_argument('-d', '--target-device',
                        default='CPU', type=str,
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('-m', '--media-type',
                        default='image', type=str,
                        choices=('image', 'video', 'webcam'),
                        help='Type of Input: image, video, cam')
    parser.add_argument('-i', '--input',
                        default='media/img/people_walking.jpg',  type=str,
                        help='Path to Input: WebCam: 0, Video File or Image file')
    parser.add_argument('-o', '--output_dir',
                        default='media/output',  type=str,
                        help='Output directory')
    parser.add_argument('-t', '--detection-threshold',
                        default=0.6,  type=float,
                        help='Object Detection Accuracy Threshold')
    parser.add_argument('--debug',
                        default=True,
                        help='Debug Mode')

    return parser.parse_args()


def get_openvino_core_net_exec(model_xml_path, model_bin_path, target_device="CPU"):
    # load IECore object
    OVIE = IECore()

    # load CPU extensions if availabel
    lib_ext_path = '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so'
    if 'CPU' in target_device and os.path.exists(lib_ext_path):
        print(f"Loading CPU extensions from {lib_ext_path}")
        OVIE.add_extension(lib_ext_path, "CPU")

    # load openVINO network
    OVNet = OVIE.read_network(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OVExec = OVIE.load_network(
        network=OVNet, device_name=target_device)

    return OVIE, OVNet, OVExec


def get_distinct_rgb_color(index):
    def hr(h):
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                  (0, 0, 0), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
                  (192, 0, 192), (0, 192, 192), (192, 192, 192), (64, 0, 0), (0, 64, 0), (0, 0, 64),
                  (64, 64, 0), (64, 0, 64), (0, 64, 64), (64, 64, 64), (32, 0, 0), (0, 32, 0),
                  (0, 0, 32), (32, 32, 0), (32, 0, 32), (0, 32, 32), (32, 32, 32), (96, 0, 0), (0, 96, 0),
                  (0, 0, 96), (96, 96, 0), (96, 0, 96), (0, 96, 96), (96, 96, 96), (160, 0, 0), (0, 160, 0),
                  (0, 0, 160), (160, 160, 0), (160, 0, 160), (0, 160, 160), (160, 160, 160), (224, 0, 0),
                  (0, 224, 0), (0, 0, 224), (224, 224, 0), (224, 0, 224), (0, 224, 224), (224, 224, 224)]
    if index >= len(color_list):
        print(
            f"WARNING:color index {index} exceeds available number of colors {len(color_list)}. Repeating colors now")
        index %= len(color_list)

    return color_list[index]
