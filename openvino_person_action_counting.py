import numpy as np
import argparse
import psutil
import time
import cv2
import os

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore

# Object Detection Labels might change based on model, so using dummy labels
object_det_labels = {i: "Fighter" for i in range(1000)}


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


def load_keras_model(model_path="models/keras_action_clsf_2/keras_model.h5"):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def get_matching_pixel_percent(patch, min_pixel, max_pixel, color_space):
    color_space_converted = cv2.cvtColor(patch, color_space)
    color_mask = cv2.inRange(color_space_converted, min_pixel, max_pixel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    color_mask = cv2.erode(color_mask, kernel, iterations=1)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    det_patch = cv2.bitwise_and(patch, patch, mask=color_mask)
    det_pixels = cv2.countNonZero(
        cv2.cvtColor(det_patch, cv2.COLOR_BGR2GRAY))
    total_pixels = patch.shape[0] * patch.shape[1]
    det_percent = det_pixels / total_pixels

    return det_percent


def inference(args) -> None:
    """
    Run Person Re-identification Application
    """
    # hyper-parameters
    MAX_DETECTIONS = 3
    MIN_BBOX_AREA_RATIO = 0.025
    BBOX_COLORS = [(0, 125, 255), (3, 244, 253)]

    # Load Person detector and Person Re-id Networks and Executables
    OVie, PDetOpenVinoNetwork, PDetOpenVinoExecutable = get_openvino_core_net_exec(
        args.pdet_model_xml, args.pdet_model_bin, args.target_device)
    ____, PReidOpenVinoNetwork, PReidOpenVinoExecutable = get_openvino_core_net_exec(
        args.preid_model_xml, args.preid_model_bin, args.target_device)

    # Get Input, Output Information
    PDetInputLayer = next(iter(PDetOpenVinoNetwork.input_info))
    PDetOutputLayer = next(iter(PDetOpenVinoExecutable.outputs))
    PReidInputLayer = next(iter(PReidOpenVinoNetwork.input_info))
    PReidOutputLayer = next(iter(PReidOpenVinoExecutable.outputs))

    if args.debug:
        print("Available Devices: ", OVie.available_devices)
        print("Person Detector Input Layer: ", PDetInputLayer)
        print("Person Detector Output Layer: ", PDetOutputLayer)
        print("Person Detector Input Shape: ",
              PDetOpenVinoNetwork.input_info[PDetInputLayer].input_data.shape)
        print("Person Detector Output Shape: ",
              PDetOpenVinoNetwork.outputs[PDetOutputLayer].shape)

        print("Person Re-identification Input Layer: ", PDetInputLayer)
        print("Person Re-identification Output Layer: ", PDetOutputLayer)
        print("Person Re-identification Input Shape: ",
              PReidOpenVinoNetwork.input_info[PReidInputLayer].input_data.shape)
        print("Person Re-identification Output Shape: ",
              PReidOpenVinoNetwork.outputs[PReidOutputLayer].shape)

    # Generate a Named Window to Show Output
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    frame_count = 0
    start_time = time.time()

    if args.media_type == 'image':
        frame_count += 1
        # Read Image
        image = cv2.imread(args.input)

        # Get Shape Values
        N, C, H, W = PDetOpenVinoNetwork.input_info[PDetInputLayer].input_data.shape

        # Pre-process Image
        resized = cv2.resize(image, (W, H))
        # Change data layout from HWC to CHW
        resized = resized.transpose((2, 0, 1))
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        start = time.time()
        results = PDetOpenVinoExecutable.infer(
            inputs={PDetInputLayer: input_image})
        end = time.time()
        inf_time = end - start
        print('Inference Time: {} Seconds Single Image'.format(inf_time))

        fps = 1. / (end - start)
        print('Estimated FPS: {} FPS Single Image'.format(fps))

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[PDetOutputLayer][0][0]
        for det in detections:
            if det[2] > args.detection_threshold:
                print('Original Frame Shape: ', fw, fh)
                xmin = int(det[3] * fw)
                ymin = int(det[4] * fh)
                xmax = int(det[5] * fw)
                ymax = int(det[6] * fh)
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymax), (0, 125, 255), 3)
                text = '{}, %: {}'.format(
                    object_det_labels[int(det[1])], round(det[2], 2))
                cv2.putText(image, text, (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

        cv2.imshow('Window', image)
        cv2.waitKey(0)
    else:
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
        pdN, pdC, pdH, pdW = PDetOpenVinoNetwork.input_info[PDetInputLayer].input_data.shape
        prN, prC, prH, prW = PReidOpenVinoNetwork.input_info[PReidInputLayer].input_data.shape
        fh, fw = frame.shape[0], frame.shape[1]
        print('Original Frame Shape: ', fw, fh)

        # reference vector for reidentification
        ref_vector0 = None
        ref_vector1 = None

        # action classification model
        # tf_action_clsf = load_keras_model()

        while has_frame:
            frame_count += 1
            resized = cv2.resize(frame, (pdW, pdH))
            resized = resized.transpose((2, 0, 1))  # HWC to CHW
            input_data = resized.reshape((pdN, pdC, pdH, pdW))
            # start inference for person detection
            results = PDetOpenVinoExecutable.infer(
                inputs={PDetInputLayer: input_data})

            fps = frame_count / (time.time() - start_time)
            inf_time = (time.time() - start_time) / frame_count
            # write fps and inference time info on frame
            text = 'FPS: {}, INF: {} ms'.format(
                round(fps, 3), round(inf_time, 3))
            cv2.putText(frame, text, (0, 22),
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (235, 229, 52), 1)
            # Print Bounding Boxes on Image
            detections = results[PDetOutputLayer][0][0]
            person_id = None
            for i, det in enumerate(detections[:MAX_DETECTIONS]):
                if det[2] > args.detection_threshold:
                    xmin = abs(int(det[3] * fw))
                    ymin = abs(int(det[4] * fh))
                    xmax = abs(int(det[5] * fw))
                    ymax = abs(int(det[6] * fh))

                    # remove background det and small bboxes
                    bbox_area = (xmax - xmin) * (ymax - ymin)
                    total_area = fw * fh
                    if bbox_area / total_area < MIN_BBOX_AREA_RATIO:
                        if args.debug:
                            print("\t Skipping since small background detected")
                        continue

                    # get person crop
                    person_crop = frame[ymin:ymax, xmin:xmax]

                    # Referee removal based on HSV color space of dark clothes
                    min_HSV, max_HSV = (0, 31, 0), (179, 199, 67)
                    referee_thres = 0.35
                    referee_color_perc = get_matching_pixel_percent(person_crop,
                                                                    min_HSV,
                                                                    max_HSV,
                                                                    color_space=cv2.COLOR_BGR2HSV)
                    if referee_color_perc >= referee_thres:
                        if args.debug:
                            print("\t Skipping since referee detected")
                        continue

                    # person re-identification
                    # preprocess crops and generate embedding vectors
                    # orig_person_crop = person_crop.copy()
                    person_crop = cv2.resize(person_crop, (prW, prH))
                    person_crop = person_crop.transpose((2, 0, 1))
                    input_data = person_crop.reshape((prN, prC, prH, prW))
                    # start re-identification inference
                    results = PReidOpenVinoExecutable.infer(
                        inputs={PReidInputLayer: input_data})
                    person_vector = results[PReidOutputLayer][0]

                    # compare vectors with previous reference vectors
                    if ref_vector0 is None:
                        ref_vector0 = person_vector
                        person_id = 0
                    elif ref_vector1 is None:
                        ref_vector1 = person_vector
                        person_id = 1
                    else:
                        if person_id is None:
                            # calculate cosine similarity between the current vec & ref vec
                            # simi = [(ref @ person_vector.T) / (np.linalg.norm(ref) * np.linalg.norm(person_vector))
                            #         for ref in (ref_vector0, ref_vector1)]
                            # person_id = simi.index(max(simi))

                            # calculate euclidean distance between the current vec & ref vec
                            dist = [np.linalg.norm(ref - person_vector)
                                    for ref in (ref_vector0, ref_vector1)]
                            person_id = dist.index(min(dist))
                        else:
                            person_id = 1 - person_id
                        # Re align reference vector
                        # uncomment when scene/objects changes drastically
                        # might cause identity switches
                        # if person_id == 0:
                        #     ref_vector0 = person_vector
                        # elif person_id == 1:
                        #     ref_vector1 = person_vector

                    # draw person bounding box
                    color = BBOX_COLORS[person_id % 2]
                    cv2.rectangle(frame,
                                  (xmin, ymin),
                                  (xmax, ymax),
                                  color, 3)
                    # draw class label
                    class_label_text = f"{object_det_labels[int(det[1])]}, {round(det[2], 4):.3f}"
                    cv2.putText(frame, class_label_text, (xmin, ymin - 7),
                                cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

                    # draw person id
                    cv2.putText(frame, f"ID: {person_id+1}", (xmin, ymin - 18),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

                    # # action classification
                    # orig_person_crop = cv2.cvtColor(cv2.resize(orig_person_crop, (224, 224)), cv2.COLOR_BGR2RGB)
                    # orig_person_crop = (orig_person_crop.astype(np.float32) / 127.0) - 1
                    # orig_person_crop = np.expand_dims(orig_person_crop, axis=0)
                    # output = tf_action_clsf.predict(orig_person_crop)[0]
                    # action = ["kick", "punch", "normal"][np.argmax(output)]
                    # # draw person action label
                    # cv2.putText(frame, f"{action}", (xmin + 60, ymin - 18),
                    #             cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 1)

            if args.debug:
                text = "SYS CPU% {} SYS MEM% {} \n " \
                       "NUM Threads {} \n " \
                       "PROC CPU% {} \n " \
                       "PROC MEM% {}".format(psutil.cpu_percent(),
                                             psutil.virtual_memory()[2],
                                             process.num_threads(),
                                             process.cpu_percent(),
                                             round(process.memory_percent(), 4))

                cv2.putText(frame, text, (0, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (250, 0, 250), 1)
            cv2.imshow('Window', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # save frame
            # cv2.imwrite(f"results/model_201/{frame_count}.jpg", frame)
            has_frame, frame = capture.read()

    end_time = time.time()
    if args.debug:
        print('Elapsed Time: {} Seconds'.format(end_time - start_time))
        print('Number of Frames: {} '.format(frame_count))
        print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Basic OpenVINO Example for Person Detection and Re-identification. Re-ID is only avai for videos')
    parser.add_argument('--preid-model-xml',
                        # default="models/person-reidentification-retail-0277-fp32/person-reidentification-retail-0277.xml",  # best model 16 fps after obj det
                        default="models/person-reidentification-retail-0286-fp32/person-reidentification-retail-0286.xml",  # balanced
                        # default="models/person-reidentification-retail-0288-fp32/person-reidentification-retail-0288.xml",  # fastest and acceptable
                        help='XML File')
    parser.add_argument('--preid-model-bin',
                        # default="models/person-reidentification-retail-0277-fp32/person-reidentification-retail-0277.bin",  # best model 16 fps after obj det
                        default="models/person-reidentification-retail-0286-fp32/person-reidentification-retail-0286.bin",  # balanced
                        # default="models/person-reidentification-retail-0288-fp32/person-reidentification-retail-0288.bin",  # fastest and acceptable
                        help='BIN File')
    parser.add_argument('--pdet-model-xml',
                        default='models/person-detection-0201/person-detection-0201.xml',
                        help='XML File')
    parser.add_argument('--pdet-model-bin',
                        default='models/person-detection-0201/person-detection-0201.bin',
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
                        default=False,
                        action='store_true',
                        help='If Flag is used, Mode is set to Debug')

    inference(parser.parse_args())
