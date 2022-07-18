import os
import queue
import tkinter as tk
from tkinter import *
import sys
from PIL import Image, ImageTk
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import datetime
import tensorflow as tf
import re
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
from skimage.draw import line
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import xlwt
import argparse
from yolo import YOLO
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')#我们可以获得当前主机上某种特定运算设备类型（如 GPU 或 CPU ）的列表
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)# 使用0号显卡，动态分配显存，有需要时申请

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test2.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.4, 'iou threshold')
flags.DEFINE_float('score', 0.55, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./weights/last.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='./outputs/caps_of_detections/parking',
                    help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')

opt = parser.parse_args()
def main(args):
    Window()


class Window():
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("研电赛违停检测平台")
        self.win.geometry('1280x720')
        self.win.resizable(True, True)
        self.text_queue = queue.Queue()
        self.EO = False

        # Buttons
        self.Button0 = tk.Button(self.win, text='Start Detection!', font=('Time New Roman', 16), bg='Green',
                                 fg='white', relief=RAISED, command=lambda: self.Start_Detection())
        self.Button0.place(x=3, y=3, width=200, height=60)

        self.Button1 = tk.Checkbutton(self.win, text='License Cognition', font=('Time New Roman', 16), bg='Yellow',
                                      fg='brown', relief=RAISED, command=lambda: self.license_plate_detection())
        self.Button1.place(x=210, y=3, width=200, height=60)

        self.Button2 = tk.Checkbutton(self.win, text='Excel Output', font=('Time New Roman', 16), bg='Yellow',
                                      fg='brown', relief=RAISED, command=lambda: self.Excel_Output())
        self.Button2.place(x=420, y=3, width=200, height=60)

        self.Button3 = tk.Button(self.win, text='Exit', font=('Time New Roman', 16), bg='white',
                                 fg='red', relief=RAISED, command=lambda: self.exit())
        self.Button3.place(x=1080, y=3, width=200, height=60)
        self.Button4 = tk.Button(self.win, text='Stop', font=('Time New Roman', 16), bg='white',
                                 fg='pink', relief=RAISED, command=lambda:self.Stop_Detection()
                                 )
        self.Button4.place(x=880, y=3, width=200, height=60)

        # Canvas
        self.canvas1 = tk.Canvas(self.win, bg='grey', width=600, height=480)
        self.canvas1.place(x=300, y=200)
        # Label
        self.Label1 = tk.Label(self.win, text='Parking Violation Detection', font=('Time New Roman', 14),
                               bg='light blue', fg='white')
        self.Label1.place(x=300, y=150, width=600, height=50)
        self.label2 = tk.Label(self.win, text='Anchor Location', font=('Time new Roman', 14),
                               bg='light blue', fg='white')
        self.label2.place(x=960, y=150, width=320, height=50)
        # text
        self.text1 = tk.Text(self.win, bg="grey", wrap=WORD)
        self.text1.place(x=960, y=200, width=320, height=480)

        self.win.mainloop()
    '''
    license_plate_detection()
    '''
    def license_plate_detection(self):
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                    self.detect()
                    create_pretrained(opt.weights, opt.weights)
            else:
                self.detect()

    '''
    get_image()用来获取图片，调整图片大小
    '''

    def get_image(self, filename, width, height):
        image = Image.open(filename).resize(width, height)
        return ImageTk.PhotoImage(image)

    '''
    Start_Detection 运行检测主线程
    '''

    def Start_Detection(self):
        self.Start = True
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = FLAGS.size
        video_path = FLAGS.video

        # load tflite model if flag is set
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # open existing workbook
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Sheet2")

        # Specifying style
        style = xlwt.easyxf('font: bold 1')
        row_num = 0

        cache_matrix = []
        viol_matrix = []
        frame_matrix = []

        # Comapare 2 regions:
        def intersection(lst1, lst2):
            return list(set(lst1) & set(lst2))

        # Check the immobility of vehicle
        def immobile(bbox, previous_bbox_str):
            previous_bbox_str = previous_bbox_str.strip()
            previous_bbox0 = int(previous_bbox_str[1:5])
            previous_bbox1 = int(previous_bbox_str[9:13])
            previous_bbox2 = int(previous_bbox_str[17:21])
            previous_bbox3 = int(previous_bbox_str[25:29])

            total = abs(bbox[0] - previous_bbox0) + abs(bbox[1] - previous_bbox1) + abs(bbox[2] - previous_bbox2) + abs(
                bbox[3] - previous_bbox3)
            if total <= 4:
                return True
            else:
                return False

        # Parking space coordinates;
        parking_co = []

        # blanked = np.zeros((658,1024), dtype=np.uint8)
        blanked = np.zeros((1920, 1280), dtype=np.uint8)
        # pts = np.array(([156, 704], [2, 893], [476, 932], [270, 708]))
        pts = np.array(([50, 200], [50, 700], [900, 100], [900, 200]))
        #pts = np.array(([100, 200], [100, 700], [900, 700], [900, 200]))  # 感兴趣区域的四个坐标
        # blanked = np.zeros((720,1280), dtype=np.uint8)
        # pts = np.array(([38, 433], [95, 322], [1246, 570], [1065, 709]))
        cv2.fillPoly(blanked, np.int32([pts]), 255)

        x_cord = np.where(blanked == 255)[1]
        y_cord = np.where(blanked == 255)[0]

        for q in range(0, len(x_cord)):
            parking_co.append((x_cord[q], y_cord[q]))

        # begin video capture
        # try:
        vid = cv2.VideoCapture('./data/video/test3.qt')
        # vid = cv2.VideoCapture('data/video/kalutara_park.mp4')
        # vid = cv2.VideoCapture('/home/hasantha/Documents/yolov4-deepsort-master/parking_violation/parking_viol3.mp4' )
        # except:
        # vid = cv2.VideoCapture(video_path)

        # out = None

        # get video ready to save locally if flag is set
        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        frame_num = 0
        # while video is running
        while self.Start==True:
            return_value, frame = vid.read()
            if return_value:

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = frame[0:1151, 365:1023]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

                # equalize the histogram of the Y channel
                frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])

                # convert the YUV image back to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # frame = cv2.resize(frame, (720, 360), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num += 1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            # allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            #allowed_classes = ['car', 'truck', 'motorbike', 'bicycle']
            allowed_classes = ['car',  'motorbike']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 1000), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2,
                            (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(bboxes, scores, names, features)]

            # initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                t = int(str(track.track_id))

                # bottom_start_co = (int(bbox[0]),int(bbox[3]))
                # bottom_stop_co = (int(bbox[2]),int(bbox[3]))
                # bbox_bottom_line_co = list(zip(*line(*bottom_start_co, *bottom_stop_co)))

                bbox_bottom_line_co = list(
                    zip(*line(*(int(bbox[0]) + 50, int(bbox[3])), *(int(bbox[2]) - 50, int(bbox[3])))))

                # b_box_co = []
                #
                # # Change accordingly
                # X, Y = np.mgrid[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
                # for m in range(0, len(X)):
                #     for n in range(0, len(X[0])):
                #         b_box_co.append((X[m][n], Y[m][n]))

                if len(intersection(parking_co, bbox_bottom_line_co)) > 0:
                    frame_matrix.append((str(frame_num) + class_name + str(t),
                                         (
                                             str(int(bbox[0])).zfill(4), str(int(bbox[1])).zfill(4),
                                             str(int(bbox[2])).zfill(4),
                                             str(int(bbox[3])).zfill(4))))
                    chk_index = str(frame_matrix).find(str(frame_num - 1) + class_name + str(t))
                    # print(frame_matrix)
                    if bool(chk_index + 1) == True:
                        previous_bbox_co_str = str(frame_matrix)[
                                               (chk_index - 1) + len(str(frame_num - 1)) + len(class_name + str(t))
                                               + 5:(chk_index - 1) + len(str(frame_num - 1)) + len(
                                                   class_name + str(t)) + 5 + 30] + " "

                        print(previous_bbox_co_str)

                        self.text_queue.put(previous_bbox_co_str+'\n')
                        self.Text_Refresh()

                        # To check the mobility
                        if immobile(bbox, previous_bbox_co_str) == True:
                            if str((class_name + str(t))) not in str(cache_matrix):
                                t_start = datetime.now()
                                cache_matrix.append((str(t_start), class_name + str(t)))
                                print(cache_matrix)

                            if (str((class_name + str(t))) in str(cache_matrix)) and (
                                    str((class_name + str(t))) not in str(viol_matrix)):

                                index = (str(cache_matrix).find(str((class_name + str(t)))))
                                t_start_cm = str(cache_matrix)[index - 28:index - 11]
                                t_spending = (datetime.now() - datetime.strptime(t_start_cm,
                                                                                 '%y-%m-%d %H:%M:%S')).total_seconds()
                                cv2.putText(frame,
                                            class_name + "-" + str(track.track_id) + "waited_time : " + str(t_spending),
                                            (int(bbox[0]), int(bbox[3] + 10)), 0, 1.5, (255, 0, 0), 2)

                                print(class_name + str(t), t_start_cm, t_spending)
                                if t_spending > 18:
                                    if self.EO == True:
                                        sheet.write(row_num, 0, str(t_start_cm), style)
                                        # sheet.write(row_num, 1, str(round(t_spending, 2)), style)
                                        sheet.write(row_num, 1, str(class_name) + str(row_num), style)
                                        row_num += 1
                                        workbook.save('outputs/xlsx/parking/details.xls')

                                    viol_matrix.append((str((class_name + str(t)))))
                                    # print(t_start_cm, t_spending, datetime.now())
                                    cropped = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                                    cropped.save(
                                        'outputs/caps_of_detections/parking/' + str(
                                            class_name) + str(row_num) + str('.jpg'))
                    else:
                        self.text_queue.put('0'+'\n')
                        self.Text_Refresh()

                        # If a vehicle in the region and being immobile for a while, started to move suddenly and stopped again, then timer will continue thanks to tracking
                        # In order to avoid that and give the vehicle another chance, this will help to remove the entry from the cache matrix and hence it will apear as a new timer
                        # else:
                        #    entry_indx = str(cache_matrix).find(class_name + str(t))
                        #    cache_matrix[cache_matrix.index(
                        #        eval(str(cache_matrix)[entry_indx - 31:entry_indx + len(class_name + str(t)) + 1]))] = None

                        # or you can simply use > cache_matrix.remove(str(cache_matrix)[entry_indx - 31:entry_indx + len(class_name + str(t)) + 1]))

                # Avoid buffer overflow
                if len(frame_matrix) > 10 ^ 7:
                    frame_matrix = []

                # print(cache_matrix)
                # print(viol_matrix)
                # print(frame_matrix)

                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[3] + 20)), 0, 1.5,
                            (255, 255, 255), 2)
                #cv2.fillPoly(frame, np.int32([pts]), 255)

                # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                        str(track.track_id),
                        class_name, (
                            int(bbox[0]),
                            int(bbox[1]),
                            int(bbox[2]),
                            int(bbox[3]))))

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            pilImage = Image.fromarray(frame)
            pilImage = pilImage.resize((600, 480), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(pilImage)
            self.canvas1.create_image(0, 0, anchor='nw', image=tkimage)

            #self.win.update()

            #if not FLAGS.dont_show:
            #    cv2.imshow("Output Video", result)

            # if output flag is set, save video file
            #if FLAGS.output:
           #     out.write(result)
            #if cv2.waitKey(1) & 0xFF == ord('q'): break
        #cv2.destroyAllWindows()


    '''
    Start_License_Detection 开启车牌检测
    '''

    '''
    Excel_Output 将检测结果输出到Excel
    '''

    def Excel_Output(self):
        self.EO = True

    def Text_Refresh(self):
        while not self.text_queue.empty():
            self.text1.insert(1.0, self.text_queue.get())
            self.text1.delete('11.0', '12.0')
            self.win.update()
    '''
    Stop_Detection()
    '''
    def Stop_Detection(self):
        self.Start = False

    def detect(save_img=False):
        yolo = YOLO()
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = True
        if classify:
            # modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
            modelc.load_state_dict(torch.load('./weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
            print("load pretrained model successful!")
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            print(pred.shape)
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred, plat_num = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for de, lic_plat in zip(det, plat_num):
                        # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                        *xyxy, conf, cls = de

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            # label = '%s %.2f' % (names[int(cls)], conf)
                            lb = ""
                            for a, i in enumerate(lic_plat):
                                # if a ==0:
                                #     continue
                                lb += CHARS[int(i)]
                            label = '%s %.2f' % (lb, conf)
                            im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            # print(type(im0))
                            img_pil = Image.fromarray(im0)  # narray转化为图片
                            im0 = yolo.detect_image(img_pil)  # 图片才能检测
                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))#不打印东西

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        im0 = np.array(im0)  # 图片转化为 narray
                        cv2.imwrite(save_path, im0)  # 这个地方的im0必须为narray

                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        im0 = np.array(im0)  # 图片转化为 narray#JMW添加
                        vid_writer.write(im0)

        if save_txt or save_img:
            # print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)


    '''
    exit（）退出程序
    '''

    def exit(self):
        sys.exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

