package com.asesorov.compar;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.LinearLayout;

public class ClassificationActivity extends Activity {
    private LinearLayout linearLayout;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Create a LinearLayout to hold the ImageViews
        linearLayout = new LinearLayout(this);
        linearLayout.setOrientation(LinearLayout.VERTICAL);
        setContentView(linearLayout);

        // Get the detected bitmaps from JNI

        // Display each bitmap as an ImageView
        /*for (Bitmap bitmap : bitmaps) {
            ImageView imageView = new ImageView(this);
            imageView.setImageBitmap(bitmap);
            linearLayout.addView(imageView);
        }*/
    }

    static {
        System.loadLibrary("yolov8ncnn");
    }
}