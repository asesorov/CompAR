# CompAR

This is my HSE diploma work: Supermarket products detection, recognition and recommendation app for Android.

Dependencies:
- https://github.com/Tencent/ncnn
- https://github.com/nihui/opencv-mobile

Models:
- YOLOv8, trained on SKU110K dataset
- ResNet50 (with lion optimizer), trained on products10k dataset
- HashModel, based on pre-trained ResNet50 features (hash size: 1024) and my own Lenta dataset

## References:
- https://github.com/FeiGeChuanShu/ncnn-android-yolov8
- https://github.com/nihui/ncnn-android-nanodet  
- https://github.com/Tencent/ncnn  
- https://github.com/ultralytics/ultralytics
- https://github.com/lucidrains/lion-pytorch
- https://products-10k.github.io/
