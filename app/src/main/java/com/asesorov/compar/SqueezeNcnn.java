package com.asesorov.compar;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class SqueezeNcnn
{
    public native boolean Init(AssetManager mgr);

    public native String Detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("squeezencnn");
    }
}