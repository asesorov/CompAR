package com.asesorov.compar;

import android.app.Activity;
import android.content.Context;
import android.widget.ScrollView;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Parcelable;
import android.widget.ImageView;
import android.view.Display;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import org.json.JSONException;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.JSONObject;
import org.json.JSONArray;

public class ClassificationActivity extends Activity {
    private TableLayout tableLayout;
    private Module module;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Bundle bundle = getIntent().getExtras();
        Parcelable[] parcelableArray = bundle.getParcelableArray("bitmaps");

        Bitmap[] bitmaps = new Bitmap[parcelableArray.length];

        for (int i = 0; i < parcelableArray.length; i++) {
            bitmaps[i] = (Bitmap) parcelableArray[i];
        }

        tableLayout = new TableLayout(this);

        ScrollView scrollView = new ScrollView(this);
        scrollView.addView(tableLayout);
        setContentView(scrollView);

        Display display = getWindowManager().getDefaultDisplay();
        int screenWidth = display.getWidth();
        int scaledWidth = screenWidth / 3;

        // Loading Classification hash model
        // Load the model
        try {
            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "hashmodel.ptl"));
        } catch (IOException e) {
            finish();
        }

        Map<String, float[][]> hashes = loadClassHashes("hashes_teacoffee_1024.json");

        for (Bitmap bitmap : bitmaps) {
            TableRow tableRow = new TableRow(this);

            ImageView imageView = new ImageView(this);
            imageView.setImageBitmap(Bitmap.createScaledBitmap(bitmap, scaledWidth, bitmap.getHeight() * scaledWidth / bitmap.getWidth(), true));
            tableRow.addView(imageView);

            TextView textView = new TextView(this);
            List<String> similarClasses = findSimilarClass(computeHash(bitmap), hashes);
            textView.setText(similarClasses.toString());
            tableRow.addView(textView);

            tableLayout.addView(tableRow);
        }
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private float[] computeHash(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
        float[] mean = {0, 0, 0};
        float[] std = {1, 1, 1};
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(scaledBitmap, mean, std);

        IValue output = module.forward(IValue.from(inputTensor));
        return output.toTensor().getDataAsFloatArray();
    }

    private Map<String, float[][]> loadClassHashes(String assetName) {
        Map<String, float[][]> classHashes = new HashMap<>();
        try {
            String jsonFilePath = assetFilePath(getApplicationContext(), assetName);
            String content = new String(Files.readAllBytes(Paths.get(jsonFilePath)));
            JSONObject json = new JSONObject(content);

            for (Iterator<String> it = json.keys(); it.hasNext(); ) {
                String key = it.next();
                JSONArray jsonArray = json.getJSONArray(key);
                float[][] hashes = new float[jsonArray.length()][];
                for (int i = 0; i < jsonArray.length(); i++) {
                    JSONArray innerArray = jsonArray.getJSONArray(i);
                    float[] innerHashes = new float[innerArray.length()];
                    for (int j = 0; j < innerArray.length(); j++) {
                        innerHashes[j] = (float) innerArray.getDouble(j);
                    }
                    hashes[i] = innerHashes;
                }
                classHashes.put(key, hashes);
            }
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }
        return classHashes;
    }


    private List<String> findSimilarClass(float[] hashVec, Map<String, float[][]> classHashes) {
        List<String> bestLabels = new ArrayList<>();
        double bestDistance = Double.POSITIVE_INFINITY;

        for (Map.Entry<String, float[][]> entry : classHashes.entrySet()) {
            String label = entry.getKey();
            float[][] hashes = entry.getValue();

            for (float[] hash : hashes) {
                double distance = euclideanDistance(hashVec, hash);
                if (distance < bestDistance) {
                    bestLabels.clear();
                    bestDistance = distance;
                    bestLabels.add(label);
                } else if (distance == bestDistance) {
                    bestLabels.add(label);
                }
            }
        }

        return bestLabels;
    }

    private double euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }


}
