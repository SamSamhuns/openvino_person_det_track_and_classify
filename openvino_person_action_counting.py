import numpy as np
import subprocess
import psutil
import time
import cv2
import os

from utils.utils import parse_arguments, get_openvino_core_net_exec, get_matching_pixel_percent, count_num_digits

# Object Detection Labels might change based on model, so using dummy labels
object_det_labels = {i: "Person" for i in range(1000)}

# frame by frame action recognition
indices_to_actions = {0: 'blue_normal',
                      1: 'blue_punch',
                      2: 'blue_kick',
                      3: 'red_normal',
                      4: 'red_punch',
                      5: 'red_kick',
                      6: 'unclear_normal',
                      7: 'unclear_punch',
                      8: 'unclear_kick',
                      9: 'unclear_unclear'}


def inference(args) -> None:
    """
    Run Person Detection and Re-identification Application
    """
    print("Running Inference for {} - {}".format(args.media_type, args.input))
    # parameters
    MAX_DETECTIONS = 3
    MIN_BBOX_AREA_RATIO = 0.025
    BBOX_COLORS = [(0, 125, 255), (3, 244, 0)]

    # create output directory
    output_dir = os.path.join(
        args.output_dir, os.path.basename(args.input).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    # Load Person detector and Person Re-id Nets and Execs
    OVie, PDetOpenVinoNet, PDetOpenVinoExec = get_openvino_core_net_exec(
        args.pdet_model_xml, args.pdet_model_bin, args.target_device)
    ____, PReidOpenVinoNet, PReidOpenVinoExec = get_openvino_core_net_exec(
        args.preid_model_xml, args.preid_model_bin, args.target_device)

    # Get Input, Output Information
    PDetInputLayer = next(iter(PDetOpenVinoNet.input_info))
    PDetOutputLayer = next(iter(PDetOpenVinoExec.outputs))
    PReidInputLayer = next(iter(PReidOpenVinoNet.input_info))
    PReidOutputLayer = next(iter(PReidOpenVinoExec.outputs))

    # action classification model
    ____, ar_net, ar_exec = get_openvino_core_net_exec(
        "models/act_clsf_openvino/action_classifier_nf_resnet50.xml",
        "models/act_clsf_openvino/action_classifier_nf_resnet50.bin")
    ar_input_layer = next(iter(ar_net.input_info))
    ar_output_layer = next(iter(ar_exec.outputs))

    if args.debug:
        # print("Available Devices: ", OVie.available_devices)
        print("Person Detector Input Layer: ", PDetInputLayer)
        print("Person Detector Output Layer: ", PDetOutputLayer)
        print("Person Detector Input Shape: ",
              PDetOpenVinoNet.input_info[PDetInputLayer].input_data.shape)
        print("Person Detector Output Shape: ",
              PDetOpenVinoNet.outputs[PDetOutputLayer].shape)

        print("Person Re-identification Input Layer: ", PDetInputLayer)
        print("Person Re-identification Output Layer: ", PDetOutputLayer)
        print("Person Re-identification Input Shape: ",
              PReidOpenVinoNet.input_info[PReidInputLayer].input_data.shape)
        print("Person Re-identification Output Shape: ",
              PReidOpenVinoNet.outputs[PReidOutputLayer].shape)

    # Generate a Named Window to Show Output
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    frame_count = 0
    start_time = time.time()

    if args.media_type == 'video':
        process_id = os.getpid()
        process = psutil.Process(process_id)

        # Implementation for CAM or Video File
        capture = cv2.VideoCapture(args.input)
        has_frame, frame = capture.read()
        frame_count += 1

        vid_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        leading_zeros = count_num_digits(vid_length + 1)
        fmt = f"0{leading_zeros}d"

        if not has_frame:
            print("Can't Open Input Video Source {}".format(args.input))
            exit(-1)

        # Get Shape Values for person det and person reid
        pdN, pdC, pdH, pdW = PDetOpenVinoNet.input_info[PDetInputLayer].input_data.shape
        prN, prC, prH, prW = PReidOpenVinoNet.input_info[PReidInputLayer].input_data.shape
        fh, fw = frame.shape[0:2]
        print('Original Frame Shape: ', fw, fh)

        # reference vector for reidentification
        ref_vector0 = None
        ref_vector1 = None

        # imagenet means and std dev
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)

        while has_frame:
            frame_count += 1
            resized = cv2.resize(frame, (pdW, pdH))
            resized = resized.transpose((2, 0, 1))  # HWC to CHW
            input_data = resized.reshape((pdN, pdC, pdH, pdW))
            # start inference for person detection
            results = PDetOpenVinoExec.infer(
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
            num_dets = 0
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
                    referee_thres = 0.4
                    referee_color_perc = get_matching_pixel_percent(person_crop,
                                                                    min_HSV,
                                                                    max_HSV,
                                                                    color_space=cv2.COLOR_BGR2HSV)
                    if referee_color_perc >= referee_thres:
                        if args.debug:
                            print("\t Skipping since referee detected")
                        continue

                    # Referee removal based on skin percent detected
                    min_HSV, max_HSV = (0, 133, 77), (235, 173, 127)
                    skin_thres = 0.008
                    skin_color_perc = get_matching_pixel_percent(person_crop,
                                                                 min_HSV,
                                                                 max_HSV,
                                                                 color_space=cv2.COLOR_BGR2HSV)
                    if skin_color_perc <= skin_thres:
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
                    results = PReidOpenVinoExec.infer(
                        inputs={PReidInputLayer: input_data})
                    person_vector = results[PReidOutputLayer][0]
                    num_dets += 1

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
                        #   uncomment when scene/objects changes drastically
                        #   might cause identity switches
                        # if person_id == 0:
                        #     ref_vector0 = person_vector
                        # elif person_id == 1:
                        #     ref_vector1 = person_vector

                    # action classification
                    orig_person_crop = frame[ymin:ymax, max(
                        xmin - 60, 0):min(xmax + 60, frame.shape[1])]
                    orig_person_crop = cv2.cvtColor(cv2.resize(
                        orig_person_crop, (380, 380)), cv2.COLOR_BGR2RGB)
                    orig_person_crop = (orig_person_crop - MEAN) / STD
                    orig_person_crop = orig_person_crop.transpose(2, 0, 1)

                    results = ar_exec.infer(
                        inputs={ar_input_layer: np.array(orig_person_crop).astype(np.float32)})
                    ac_out = results[ar_output_layer][0]
                    action = indices_to_actions[np.argmax(ac_out)]
                    overrite_color = (0, 0, 255) if 'red' in action else (
                        255, 0, 0) if 'blue' in action else None
                    # draw person action label if not normal
                    if action != "normal":
                        cv2.putText(frame, f"{action}", (xmin + 60, ymin - 18),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 0), 1)

                    # draw person bounding box
                    color = BBOX_COLORS[person_id %
                                        2] if overrite_color is None else overrite_color
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

            if num_dets < 2 and frame_count < 100:
                # reset the detection vectors if less than one person detected
                ref_vector0 = None
                ref_vector1 = None

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

            cv2.imwrite(os.path.join(output_dir, f"frame_%{fmt}.jpg" % (frame_count)),
                        frame)
            has_frame, frame = capture.read()

        # convert saved frames into video
        command = ["ffmpeg",
                   "-r", f"{20}",
                   "-start_number", "0" * (leading_zeros),
                   "-i", f"{output_dir}/frame_%{leading_zeros}d.jpg",
                   "-vcodec", "libx264",
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   "-y", "-an", f"{output_dir}/{args.input.split('/')[-2]}.mp4"]
        output, error = subprocess.Popen(
            command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    else:
        print(f"{args.media_type} media_type is not recognized")
        print("Only video are allowed. Use -h for help")

    end_time = time.time()
    print('Elapsed Time: {} Seconds'.format(end_time - start_time))
    print('Number of Frames: {} '.format(frame_count))
    print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


if __name__ == '__main__':
    args = parse_arguments(
        desc="Basic OpenVINO Example for person action counting")
    inference(args)
