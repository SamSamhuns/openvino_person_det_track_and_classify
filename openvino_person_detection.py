import psutil
import time
import cv2
import os

from utils.utils import parse_arguments, get_openvino_core_net_exec

# dummy labels
OBJECT_DET_LABELS = {i: "Object" for i in range(1000)}


def inference_image(image_ndarray, OVDesc, process, args):
    """
    returns image as np array with dets drawn on it
    OVDesc is a tuple of (OVNet, OVExec, InputLayer, OutputLayer)
    """
    OVNet, OVExec, InputLayer, OutputLayer = OVDesc
    N, C, H, W = OVNet.input_info[InputLayer].input_data.shape
    resized = cv2.resize(image_ndarray, (W, H))
    resized = resized.transpose((2, 0, 1))  # HWC to CHW
    input_image = resized.reshape((N, C, H, W))

    # Start Inference
    start = time.time()
    results = OVExec.infer(inputs={InputLayer: input_image})
    end = time.time()
    inf_time = end - start
    print('Inference Time: {} Seconds Single Image'.format(inf_time))

    fps = 1. / (end - start)
    print('Estimated FPS: {} FPS Single Image'.format(fps))

    fh, fw = image_ndarray.shape[0:2]
    # Write fos, inference info on Image
    text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
    cv2.putText(image_ndarray, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                0.6, (0, 125, 255), 1)

    # Print Bounding Boxes on Image
    detections = results[OutputLayer][0][0]
    for detection in detections:
        if detection[2] > args.detection_threshold:
            xmin = int(detection[3] * fw)
            ymin = int(detection[4] * fh)
            xmax = int(detection[5] * fw)
            ymax = int(detection[6] * fh)
            cv2.rectangle(image_ndarray, (xmin, ymin),
                          (xmax, ymax), (0, 125, 255), 3)
            text = '{}, %: {}'.format(
                OBJECT_DET_LABELS[int(detection[1])], round(detection[2], 2))
            cv2.putText(image_ndarray, text, (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

    proc_text = "SYS CPU% {} SYS MEM% {} \n " \
        "NUM Threads {} \n " \
        "PROC CPU% {} \n " \
        "PROC MEM% {}".format(psutil.cpu_percent(),
                              psutil.virtual_memory()[2],
                              process.num_threads(),
                              process.cpu_percent(),
                              round(process.memory_percent(), 4))

    cv2.putText(image_ndarray, proc_text, (0, 50),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 250), 1)
    return image_ndarray


def inference(args) -> None:
    """
    Run Object Detection Application
    """
    print("Running Inference for {}: {}".format(args.media_type, args.input))
    # Load Network and Executable
    OVIE, OVNet, OVExec = get_openvino_core_net_exec(
        args.pdet_model_xml, args.pdet_model_bin, args.target_device)

    # Get Input, Output Information
    InputLayer = next(iter(OVNet.input_info))
    OutputLayer = next(iter(OVNet.outputs))

    # create output directory
    output_dir = os.path.join(
        args.output_dir, os.path.basename(args.input).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    if args.debug:
        print("Available Devices: ", OVIE.available_devices)
        print("Input Layer: ", InputLayer)
        print("Output Layer: ", OutputLayer)
        print("Input Shape: ",
              OVNet.input_info[InputLayer].input_data.shape)
        print("Output Shape: ", OVNet.outputs[OutputLayer].shape)

    # Generate a Named Window to Show Output
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    # get process and system info
    process_id = os.getpid()
    process = psutil.Process(process_id)
    # get OV Desc tuple object
    OVDesc = (OVNet, OVExec, InputLayer, OutputLayer)

    frame_count = 0
    start_time = time.time()
    if args.media_type == 'image':
        frame_count += 1
        frame = cv2.imread(args.input)
        image = inference_image(frame, OVDesc, process, args)
        cv2.imwrite(os.path.join(output_dir, f"{frame_count}.jpg"), image)
        cv2.imshow('Window', image)
        cv2.waitKey(0)
    elif args.media_type in {'video', 'webcam'}:
        # Implementation for CAM or Video File
        vid_src = args.input if args.media_type == "video" else 0
        capture = cv2.VideoCapture(vid_src)
        has_frame, frame = capture.read()
        frame_count += 1

        if not has_frame:
            print("Can't Open Input Video Source {}".format(args.input))
            exit(-1)

        while has_frame:
            frame_count += 1
            image = inference_image(frame, OVDesc, process, args)
            cv2.imwrite(os.path.join(output_dir, f"{frame_count}.jpg"), image)
            cv2.imshow('Window', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            has_frame, frame = capture.read()
    else:
        print(f"{args.media_type} media_type is not recognized")
        print("Only image/video/webcam are allowed. Use -h for help")

    end_time = time.time()
    if args.debug:
        print('Elapsed Time: {} Seconds'.format(end_time - start_time))
        print('Number of Frames: {} '.format(frame_count))
        print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


if __name__ == '__main__':
    args = parse_arguments(
        desc="Basic OpenVINO Example for person/object detection")
    inference(args)
