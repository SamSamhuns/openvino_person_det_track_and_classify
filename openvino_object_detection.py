import cv2 as cv
import argparse
import psutil
import time
import os

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore


# might be different for different models
# object_det_labels = {0: "person",
#                      1: "person"}
object_det_labels = {i: "Object" for i in range(1000)}


def get_openvino_core_net_exec(model_xml_path, model_bin_path, target_device="CPU"):
    # load IECore object
    OpenVinoIE = IECore()

    # load CPU extensions if necessary
    # if 'CPU' in args.target_device:
    #     OpenVinoIE.add_extension(
    #         '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so', "CPU")

    # load openVINO network
    OpenVinoNetwork = OpenVinoIE.read_network(
        model=model_xml_path, weights=model_bin_path)

    # create executable network
    OpenVinoExecutable = OpenVinoIE.load_network(
        network=OpenVinoNetwork, device_name=target_device)

    return OpenVinoIE, OpenVinoNetwork, OpenVinoExecutable


def inference(args):
    """
    Run Object Detection Application
    :return:
    """
    frame_count = 0
    # Load Network and Executable
    OpenVinoIE, OpenVinoNetwork, OpenVinoExecutable = get_openvino_core_net_exec(
        args.model_xml, args.model_bin, args.target_device)

    # Get Input, Output Information
    InputLayer = next(iter(OpenVinoNetwork.input_info))
    OutputLayer = next(iter(OpenVinoNetwork.outputs))

    if args.debug:
        print("Available Devices: ", OpenVinoIE.available_devices)
        print("Input Layer: ", InputLayer)
        print("Output Layer: ", OutputLayer)
        print("Input Shape: ",
              OpenVinoNetwork.input_info[InputLayer].input_data.shape)
        print("Output Shape: ", OpenVinoNetwork.outputs[OutputLayer].shape)

    # Generate a Named Window to Show Output
    cv.namedWindow('Window', cv.WINDOW_NORMAL)
    cv.resizeWindow('Window', 800, 600)

    start_time = time.time()

    if args.media_type == 'image':
        frame_count += 1
        # Read Image
        image = cv.imread(args.input)

        # Get Shape Values
        N, C, H, W = OpenVinoNetwork.input_info[InputLayer].input_data.shape

        # Pre-process Image
        resized = cv.resize(image, (W, H))
        # Change data layout from HWC to CHW
        resized = resized.transpose((2, 0, 1))
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        start = time.time()
        results = OpenVinoExecutable.infer(inputs={InputLayer: input_image})
        end = time.time()
        inf_time = end - start
        print('Inference Time: {} Seconds Single Image'.format(inf_time))

        fps = 1. / (end - start)
        print('Estimated FPS: {} FPS Single Image'.format(fps))

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv.putText(image, text, (0, 20), cv.FONT_HERSHEY_COMPLEX,
                   0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[OutputLayer][0][0]
        for detection in detections:
            if detection[2] > args.detection_threshold:
                print('Original Frame Shape: ', fw, fh)
                xmin = int(detection[3] * fw)
                ymin = int(detection[4] * fh)
                xmax = int(detection[5] * fw)
                ymax = int(detection[6] * fh)
                cv.rectangle(image, (xmin, ymin),
                             (xmax, ymax), (0, 125, 255), 3)
                text = '{}, %: {}'.format(
                    object_det_labels[int(detection[1])], round(detection[2], 2))
                cv.putText(image, text, (xmin, ymin - 7),
                           cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

        cv.imshow('Window', image)
        cv.waitKey(0)
    else:
        print("Running Inference for {} - {}".format(args.media_type, args.input))
        process_id = os.getpid()
        process = psutil.Process(process_id)

        # Implementation for CAM or Video File
        capture = cv.VideoCapture(args.input)
        has_frame, frame = capture.read()
        frame_count += 1

        if not has_frame:
            print("Can't Open Input Video Source {}".format(args.input))
            exit(-1)

        # Get Shape Values
        N, C, H, W = OpenVinoNetwork.inputs[InputLayer].shape
        fh = frame.shape[0]
        fw = frame.shape[1]
        print('Original Frame Shape: ', fw, fh)

        while has_frame:
            frame_count += 1
            resized = cv.resize(frame, (W, H))
            # Change data layout from HWC to CHW
            resized = resized.transpose((2, 0, 1))
            input_data = resized.reshape((N, C, H, W))
            # Start Inference
            results = OpenVinoExecutable.infer(
                inputs={InputLayer: input_data})

            fps = frame_count / (time.time() - start_time)
            inf_time = (time.time() - start_time) / frame_count
            # Write Information on Image
            text = 'FPS: {}, INF: {} ms'.format(
                round(fps, 3), round(inf_time, 3))
            cv.putText(frame, text, (0, 20),
                       cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 125, 255), 1)

            # Print Bounding Boxes on Image
            detections = results[OutputLayer][0][0]
            for detection in detections:
                if detection[2] > args.detection_threshold:
                    xmin = int(detection[3] * fw)
                    ymin = int(detection[4] * fh)
                    xmax = int(detection[5] * fw)
                    ymax = int(detection[6] * fh)
                    cv.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), (0, 125, 255), 3)
                    detection_percentage = round(detection[2], 4)
                    text = '{}, %: {}'.format(
                        object_det_labels[int(detection[1])], detection_percentage)
                    cv.putText(frame, text, (xmin, ymin - 7),
                               cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

            text = "SYS CPU% {} SYS MEM% {} \n " \
                   "NUM Threads {} \n " \
                   "PROC CPU% {} \n " \
                   "PROC MEM% {}".format(psutil.cpu_percent(),
                                         psutil.virtual_memory()[2],
                                         process.num_threads(),
                                         process.cpu_percent(),
                                         round(process.memory_percent(), 4))

            cv.putText(frame, text, (0, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 250), 1)
            cv.imshow('Window', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            # save frame
            # cv.imwrite(f"results/model_201/{frame_count}.jpg", frame)
            has_frame, frame = capture.read()

    end_time = time.time()
    if args.debug:
        print('Elapsed Time: {} Seconds'.format(end_time - start_time))
        print('Number of Frames: {} '.format(frame_count))
        print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Basic OpenVINO Example for object detection')
    # all FPS recorded on the media/vid/ufc_sample.mp4 video
    parser.add_argument('--model-xml',
                        # default='models/person-detection-retail-0013/person-detection-retail-0013.xml',  # 31 fps
                        # default='models/person-detection-0200/person-detection-0200.xml',                # 37 fps best & fastest
                        # default='models/person-detection-0200-int8/person-detection-0200.xml',           # 37 fps
                        default='models/person-detection-0201/person-detection-0201.xml',                # 30 fps
                        # default='models/person-detection-0202/person-detection-0202.xml',                # 23 FPS
                        help='XML File')
    parser.add_argument('--model-bin',
                        # default='models/person-detection-retail-0013/person-detection-retail-0013.bin',  # 31 fps
                        # default='models/person-detection-0200/person-detection-0200.bin',                # 37 fps best & fastest
                        # default='models/person-detection-0200-int8/person-detection-0200.bin',           # 37 fps
                        default='models/person-detection-0201/person-detection-0201.bin',                # 30 fps
                        # default='models/person-detection-0202/person-detection-0202.bin',                # 23 FPS
                        help='BIN File')
    parser.add_argument('-t',
                        '--target-device',
                        default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('-m',
                        '--media-type',
                        default='image',
                        help='Type of Input: image, video, cam')
    parser.add_argument('-i',
                        '--input',
                        default='media/img/people_walking.jpg',
                        help='Path to Input: WebCam: 0, Video File or Image file')
    parser.add_argument('-d',
                        '--detection-threshold',
                        default=0.6,
                        help='Object Detection Accuracy Threshold')
    parser.add_argument('--debug',
                        default=True,
                        help='Debug Mode')

    inference(parser.parse_args())
