![CompAR](images/CompAR.png?raw=true "CompAR")

This is my HSE diploma work: Supermarket products detection, recognition and recommendation app for Android.

Dependencies:
- https://github.com/Tencent/ncnn
- https://github.com/nihui/opencv-mobile

Models:
- YOLOv8, trained on SKU110K dataset (75% acc)
- MobileNet-v3 (with lion optimizer), trained on RP2K dataset (93% acc)
- ProductHash, my own model based on pre-trained MobileNet-v3 features (hash size: 256) and my own Lenta dataset (~100% acc)

## References:
- https://github.com/FeiGeChuanShu/ncnn-android-yolov8
- https://github.com/nihui/ncnn-android-nanodet  
- https://github.com/Tencent/ncnn  
- https://github.com/ultralytics/ultralytics
- https://github.com/lucidrains/lion-pytorch
- https://www.pinlandata.com/rp2k_dataset/
- https://arxiv.org/pdf/2006.12634v7.pdf
