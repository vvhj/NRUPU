"""
@Author: Dongjingdian Liu
@Date: 2020-08-01 12:36:02
@LastEditTime: 2020-08-01 12:36:03
@Description: models.py
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Boolean

Base = declarative_base()


# init when init system update when submit device settings
class Device(Base):
    __tablename__ = "device"
    id = Column(Integer, primary_key=True)
    name = Column(String, default="摄像头")
    ip = Column(String, default="0.0.0.0")
    type = Column(String, default="camera")
    username = Column(String, default="admin")
    password = Column(String, default="12345")


# Foreign object detection
# filled when submit device settings
class FodCfg(Base):
    __tablename__ = "fodCfg"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    virtual_gpu_id = Column(Integer)
    n_warning_threshold = Column(Integer, default=10000)
    ex_warning_threshold = Column(Integer, default=40000)

class DevCfg(Base):
    __tablename__ = "devCfg"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    virtual_gpu_id = Column(Integer)
    l_warning_threshold = Column(Integer, default=0.95)
    r_warning_threshold = Column(Integer, default=0.95)

class FodRecord(Base):
    __tablename__ = "fodRecord"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    device_id = Column(Integer)
    dnn_model_id = Column(Integer, default=1)
    status = Column(String)
    storage_path = Column(String)
    tags = Column(String, default="")
    areas = Column(String, default="")
    location = Column(String, default="无")

class DevRecord(Base):
    __tablename__ = "devRecord"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True)
    device_id = Column(Integer)
    dnn_model_id = Column(Integer, default=1)
    status = Column(String)
    storage_path = Column(String)
    tags = Column(String, default="")
    #areas = Column(String, default="")
    location = Column(String, default="无")


# Belt deviation detection
# filled when submit device settings
class BddCfg(Base):
    __tablename__ = "bddCfg"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    offset_distance = Column(Integer)


# filled when init system
# class ModeCategory(Base):
#     __tablename__ = "modeCategory"
#     id = Column(Integer, primary_key=True)
#     name = Column(String)


# filled when submit device settings
# all dnn models include fod, bdd and others
class DnnModel(Base):
    __tablename__ = "dnnModel"
    id = Column(Integer, primary_key=True)
    category = Column(String)
    classes = Column(String)  # use whitespace to split class names
    weight = Column(String)


# filled when init system
class VirtualGpu(Base):
    __tablename__ = "virtualGpu"
    id = Column(Integer, primary_key=True)
    gpu_id = Column(Integer)
    used = Column(Boolean)


# filled when submit device settings
class DeviceLocation(Base):
    __tablename__ = "deviceLocation"
    id = Column(Integer, primary_key=True)
    device_id = Column(Integer)
    location = Column(String)


# system status(running training or erroring...)
class SystemStatus(Base):
    __tablename__ = "systemStatus"
    id = Column(Integer, primary_key=True)
    status = Column(String, default="running")
