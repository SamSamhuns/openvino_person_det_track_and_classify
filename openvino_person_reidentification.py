import numpy as np
import psutil
import time
import cv2
import os

from utils.utils import parse_arguments, get_openvino_core_net_exec, get_distinct_rgb_color

# Object Detection Labels might change based on model, so using dummy labels
object_det_labels = {i: "Object" for i in range(1000)}


def inference(args):
    """
    Run Person Re-identification Application
    :return:
    """
    print("Running Inference for {}: {}".format(args.media_type, args.input))
    MAX_DETECTIONS = 3  # max number of bounding boxes considered
    MAX_TO_TRACK = 3    # max number of bounding boxes to track
    MIN_BBOX_AREA_RATIO = 0.025

    # create output directory
    output_dir = os.path.join(
        args.output_dir, os.path.basename(args.input).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    # Load Person detector and Person Re-id Networks and Executables
    OVie, PDetOVNet, PDetOVExec = get_openvino_core_net_exec(
        args.pdet_model_xml, args.pdet_model_bin, args.target_device)
    ____, PReidOVNet, PReidOVExec = get_openvino_core_net_exec(
        args.preid_model_xml, args.preid_model_bin, args.target_device)

    # Get Input, Output Information
    PDetInputLayer = next(iter(PDetOVNet.input_info))
    PDetOutputLayer = next(iter(PDetOVExec.outputs))
    PReidInputLayer = next(iter(PReidOVNet.input_info))
    PReidOutputLayer = next(iter(PReidOVExec.outputs))

    if args.debug:
        print("Available Devices: ", OVie.available_devices)
        print("Person Detector Input Layer: ", PDetInputLayer)
        print("Person Detector Output Layer: ", PDetOutputLayer)
        print("Person Detector Input Shape: ",
              PDetOVNet.input_info[PDetInputLayer].input_data.shape)
        print("Person Detector Output Shape: ",
              PDetOVNet.outputs[PDetOutputLayer].shape)

        print("Person Re-identification Input Layer: ", PDetInputLayer)
        print("Person Re-identification Output Layer: ", PDetOutputLayer)
        print("Person Re-identification Input Shape: ",
              PReidOVNet.input_info[PReidInputLayer].input_data.shape)
        print("Person Re-identification Output Shape: ",
              PReidOVNet.outputs[PReidOutputLayer].shape)

    # Generate a Named Window to Show Output
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    frame_count = 0
    start_time = time.time()
    if args.media_type in {'video', 'webcam'}:
        print("Running Inference for {} - {}".format(args.media_type, args.input))
        process_id = os.getpid()
        process = psutil.Process(process_id)

        # Implementation for CAM or Video File
        capture = cv2.VideoCapture(args.input)
        has_frame, frame = capture.read()
        frame_count += 1

        if not has_frame:
            print("Can't Open Input Video Source {}".format(args.input))
            exit(-1)

        # Get Shape Values for person det and person reid
        pdN, pdC, pdH, pdW = PDetOVNet.input_info[PDetInputLayer].input_data.shape
        prN, prC, prH, prW = PReidOVNet.input_info[PReidInputLayer].input_data.shape
        fh, fw = frame.shape[0:2]
        print('Original Frame Shape: ', fw, fh)

        # reference vector for reidentification
        ref_vector_list = [None for _ in range(MAX_TO_TRACK)]

        while has_frame:
            frame_count += 1
            resized = cv2.resize(frame, (pdW, pdH))
            resized = resized.transpose((2, 0, 1))  # HWC to CHW
            input_data = resized.reshape((pdN, pdC, pdH, pdW))
            # start inference for person detection
            results = PDetOVExec.infer(
                inputs={PDetInputLayer: input_data})

            fps = frame_count / (time.time() - start_time)
            inf_time = (time.time() - start_time) / frame_count
            # Write Information on Image
            text = 'FPS: {}, INF: {} ms'.format(
                round(fps, 3), round(inf_time, 3))
            cv2.putText(frame, text, (0, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 125, 255), 1)

            # Print Bounding Boxes on Image
            detections = results[PDetOutputLayer][0][0]
            for det in detections[:MAX_DETECTIONS]:
                if det[2] > args.detection_threshold:
                    xmin = abs(int(det[3] * fw))
                    ymin = abs(int(det[4] * fh))
                    xmax = abs(int(det[5] * fw))
                    ymax = abs(int(det[6] * fh))

                    # remove background det and small bboxes
                    bbox_area = (xmax - xmin) * (ymax - ymin)
                    total_area = fw * fh
                    if bbox_area / total_area < MIN_BBOX_AREA_RATIO:
                        # print("\t Skipping since small background detected")
                        continue

                    # get person crop
                    person_crop = frame[ymin:ymax, xmin:xmax]
                    # person re-identification
                    person_crop = cv2.resize(person_crop, (prW, prH))
                    person_crop = person_crop.transpose((2, 0, 1))  # HWC 2 CHW
                    input_data = person_crop.reshape((prN, prC, prH, prW))
                    # start re-identification inference
                    results = PReidOVExec.infer(
                        inputs={PReidInputLayer: input_data})
                    person_vector = results[PReidOutputLayer][0]

                    # compare vectors with previous reference vectors
                    person_id = 0
                    calc_distance = True
                    for i, ref in enumerate(ref_vector_list):
                        if ref is None:  # ideally should run only for the first frame
                            ref_vector_list[i] = person_vector
                            calc_distance = False
                            person_id = i + 1
                            break
                    if calc_distance:
                        dist = [(ref @ person_vector.T) / (np.linalg.norm(ref) * np.linalg.norm(person_vector))
                                for ref in ref_vector_list]
                        person_id = dist.index(min(dist)) + 1
                        # re-align reference vector
                        #   uncomment when scene/objects changes drastically
                        #   might cause identity switches if uncommented
                        # ref_vector_list[person_id - 1] = person_vector
                    pid_color = get_distinct_rgb_color(person_id)

                    cv2.putText(frame, f"ID: {person_id}", (xmin, ymin - 18),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, pid_color, 1)

                    # draw bounding box on image
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  pid_color, 3)
                    det_conf = round(det[2], 4)
                    text = f"{object_det_labels[int(det[1])]}, {det_conf:.3f}"
                    cv2.putText(frame, text, (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, pid_color, 1)

            proc_text = "SYS CPU% {} SYS MEM% {} \n " \
                "NUM Threads {} \n " \
                "PROC CPU% {} \n " \
                "PROC MEM% {}".format(psutil.cpu_percent(),
                                      psutil.virtual_memory()[2],
                                      process.num_threads(),
                                      process.cpu_percent(),
                                      round(process.memory_percent(), 4))
            cv2.putText(frame, proc_text, (0, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 250), 1)

            cv2.imshow('Window', frame)
            cv2.imwrite(os.path.join(output_dir, f"{frame_count}.jpg"), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            has_frame, frame = capture.read()
    else:
        print(f"{args.media_type} media_type is not recognized")
        print("Only video/webcam are allowed. Use -h for help")

    end_time = time.time()
    if args.debug:
        print('Elapsed Time: {} Seconds'.format(end_time - start_time))
        print('Number of Frames: {} '.format(frame_count))
        print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


if __name__ == '__main__':
    args = parse_arguments(
        desc="Basic OpenVINO Example for person re-idenfication")
    inference(args)
