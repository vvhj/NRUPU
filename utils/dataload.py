import cv2
import os

from time import time
from pymemcache import serde
from pymemcache.client.base import Client
import torchvision.transforms as transforms
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone, timedelta
from multiprocessing import Process
from utils.models import DevRecord,DeviceLocation,DevCfg
import hashlib
import logging
engine = create_engine("postgresql://quantum:429526000@postgres/yqdb")
Session = sessionmaker(bind=engine)
session = Session()

def call_alarm(location):
    os.chdir("/deviation/Alarm_lib/linux64/lib")
    location = hashlib.md5(location.encode("utf-8")).digest().hex()
    if status_register.get(f"{location}_alarm_status") == "empty":
        status_register.set(f"{location}_alarm_status", "runing")
        if location == hashlib.md5(("丈八采区").encode("utf-8")).digest().hex():
            logging.warning("丈八采区 alarming...")
            os.system("./8th_mining_area")
        elif location == hashlib.md5(("十四采区").encode("utf-8")).digest().hex():
            logging.warning("十四采区 alarming...")
            os.system("./14th_mining_area")
        status_register.set(f"{location}_alarm_status", "empty")

img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

status_register = Client(
    ("status_register", 12001),
    serializer=serde.python_memcache_serializer,
    deserializer=serde.python_memcache_deserializer,
)

fod_image_register_A = Client(
    ("fod_image_register_A", 12004),
    serializer=serde.python_memcache_serializer,
    deserializer=serde.python_memcache_deserializer,
)

fod_image_register_B = Client(
    ("fod_image_register_B", 12005),
    serializer=serde.python_memcache_serializer,
    deserializer=serde.python_memcache_deserializer,
)

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)

class LoadImages:  # for inference
    def __init__(self, img_size=640):
        self.img_size = img_size
        temp_img = cv2.imread("demo/loading.png")
        for i in range(1, 5):
            status_register.set(f"fod_pipeline_{i}_time", time())
            status_register.set(f"fod_pipeline_{i}_max_area", 1)
            status_register.set(f"fod_pipeline_{i}_core_y", 1)
            fod_image_register_A.set(f"fod_pipeline_{i}", temp_img)

    def __iter__(self):
        self.pipeline = 0
        return self

    def __next__(self):
        self.pipeline = self.pipeline + 1 if self.pipeline < 4 else 1
        img0 = fod_image_register_A.get(f"fod_pipeline_{self.pipeline}")

        # Padded resize
        # img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return self.pipeline, img, img0

def send_img(pipeline, img):
    fod_image_register_B.set(f"fod_pipeline_{pipeline}", img)

def trigger_alarm(pipeline):
    device_id = (
        session.query(DevCfg).filter_by(virtual_gpu_id=pipeline).first().device_id
    )
    location = (
        session.query(DeviceLocation).filter_by(device_id=device_id).first().location
    )
    alarm_process = Process(target=call_alarm, args=(location,),)
    alarm_process.start()

def save_img(pipeline, img, status):
    device_id = (
        session.query(DevCfg).filter_by(virtual_gpu_id=pipeline).first().device_id
    )
    if time() - status_register.get(f"fod_pipeline_{pipeline}_time") > 1:
        status_register.set(f"fod_pipeline_{pipeline}_time", time())
        timestamp = datetime.utcnow().astimezone(timezone(timedelta(hours=8)))
        file_path = Path("utils/blank_image.jpg")
        dir_path = Path("/yolov5/photos").joinpath(
            timestamp.strftime("%Y"),
            timestamp.strftime("%B"),
            timestamp.strftime("%d"),
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path.joinpath(timestamp.strftime(r"%Y%m%d%H%M%S%f") + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), img)

        fod_record = DevRecord(
            device_id=device_id,
            status=status,
            timestamp=timestamp,
            storage_path=str(file_path),
            location=session.query(DeviceLocation)
            .filter_by(device_id=device_id)
            .first()
            .location,
        )
        session.add(fod_record)
        session.commit()

def init_cache():
    for i in range(1, 5):
        status_register.set(f"Dev_pipeline_{i+1}_lThreshold", 0.95)
        status_register.set(f"Dev_pipeline_{i+1}_rThreshold", 0.95)
    for i, cfg in enumerate(
        session.query(DevCfg).order_by(DevCfg.virtual_gpu_id).all()
    ):
        # if cfg.l_warning_threshold < 5000:
        #     cfg.l_warning_threshold = 12500
        # if cfg.r_warning_threshold < 5000:
        #     cfg.r_warning_threshold = 18000
        status_register.set(f"Dev_pipeline_{i+1}_lThreshold", cfg.l_warning_threshold)
        status_register.set(f"Dev_pipeline_{i+1}_rxThreshold", cfg.r_warning_threshold)
