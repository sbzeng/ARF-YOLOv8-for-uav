1，Add the yaml file in the model folder to the YOLOv8 official library;
2，Use the metrics.py in sa-CIoU folder to replace the ultralytics/ultralytics/utils metrics.py in the YOLOv8 official;
3, Add the contents of the CA2f folder to ultralytics/ultralytics/nn/modules blocks.pt;
4, After completing steps 1 to 3 and successfully installing the YOLOv8 library, you can execute training using the command "yolo task=detect mode=train model= "your model path" data=data=datasets/VisDrone.yaml batch=4 epochs=300 imgsz=640".
ARF-YOLOv8 is written based on the official YOLOv8 library and is used for target detection in UAV aerial images. The URL of the official YOLOv8 library is:https://github.com/ultralytics/ultralytics/tree/main.
