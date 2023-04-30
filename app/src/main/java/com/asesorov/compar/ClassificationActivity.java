package com.asesorov.compar;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.text.TextUtils;
import android.view.View;
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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.JSONObject;

import android.view.ViewGroup;
import android.view.LayoutInflater;

import androidx.annotation.NonNull;
import androidx.core.util.Pair;
import androidx.viewpager.widget.PagerAdapter;
import androidx.viewpager.widget.ViewPager;

import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

public class ClassificationActivity extends Activity {
    private Module module;
    private JSONObject productData;

    // Release resources and memory
    private void releaseResources() {
        if (module != null) {
            module = null;
        }
        if (productData != null) {
            productData = null;
        }
        System.gc();
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classification);

        Bundle bundle = getIntent().getExtras();
        Parcelable[] parcelableArray = bundle.getParcelableArray("bitmaps");

        Bitmap[] bitmaps = new Bitmap[parcelableArray.length];

        for (int i = 0; i < parcelableArray.length; i++) {
            bitmaps[i] = (Bitmap) parcelableArray[i];
        }

        // Load product data from assets
        productData = loadProductData("ssjd.json");

        // Loading Classification hash model
        try {
            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "hashmodelmobilenetv3large256sign.ptl"));
        } catch (IOException e) {
            finish();
        }

        Map<String, float[][]> hashes = loadClassHashes("hashes_lenta_256.json");

        // Compute hashes for input bitmaps
        float[][] inputHashes = new float[bitmaps.length][];
        for (int i = 0; i < bitmaps.length; i++) {
            inputHashes[i] = computeHash(bitmaps[i]);
        }

        // Perform k-means clustering on input hashes
        Pair<float[][], int[]> kMeansResult = kMeans(inputHashes, 300); // 300 is the maximum number of iterations
        float[][] centroids = kMeansResult.first;
        int[] assignments = kMeansResult.second;

        // Create a map to store the representative bitmap for each cluster
        Map<Integer, Bitmap> representativeBitmaps = new HashMap<>();
        for (int i = 0; i < assignments.length; i++) {
            int cluster = assignments[i];
            if (!representativeBitmaps.containsKey(cluster)) {
                representativeBitmaps.put(cluster, bitmaps[i]);
            }
        }

        ArrayList<JSONObject> products = new ArrayList<>();
        for (float[] centroid : centroids) {
            String productId = findSimilarClass(centroid, hashes);

            try {
                JSONObject productInfo = productData.getJSONObject(productId);
                productInfo.put("id", productId);
                products.add(productInfo);
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        // Release the resources after the classification is done
        releaseResources();

        // Create an array of representative bitmaps in the same order as the products
        Bitmap[] representativeBitmapArray = new Bitmap[products.size()];
        int bitmapIndex = 0;
        for (int cluster : representativeBitmaps.keySet()) {
            representativeBitmapArray[bitmapIndex] = representativeBitmaps.get(cluster);
            bitmapIndex++;
        }

        ViewPager viewPager = findViewById(R.id.view_pager);
        viewPager.setAdapter(new CustomPagerAdapter(this, products, representativeBitmapArray));
    }

    private JSONObject loadProductData(String assetName) {
        JSONObject productData = new JSONObject();

        try {
            String jsonFilePath = assetFilePath(getApplicationContext(), assetName);
            String jsonString = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                jsonString = new String(Files.readAllBytes(Paths.get(jsonFilePath)));
            }
            productData = new JSONObject(jsonString);
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }

        return productData;
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
        float[] mean = {.485f, .456f, .406f};
        float[] std = {.229f, .224f, .225f};
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(scaledBitmap, mean, std);

        IValue output = module.forward(IValue.from(inputTensor));
        return output.toTensor().getDataAsFloatArray();
    }

    private Map<String, float[][]> loadClassHashes(String assetName) {
        Map<String, float[][]> classHashes = new HashMap<>();
        Gson gson = new Gson();

        try {
            String jsonFilePath = assetFilePath(getApplicationContext(), assetName);
            InputStreamReader inputStreamReader = new InputStreamReader(new FileInputStream(jsonFilePath));
            JsonReader jsonReader = new JsonReader(inputStreamReader);

            jsonReader.beginObject();
            while (jsonReader.hasNext()) {
                String key = jsonReader.nextName();
                float[][] hashes = gson.fromJson(jsonReader, float[][].class);
                classHashes.put(key, hashes);
            }
            jsonReader.endObject();
            jsonReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return classHashes;
    }

    private String findSimilarClass(float[] hashVec, Map<String, float[][]> classHashes) {
        String bestLabel = null;
        double bestDistance = Double.POSITIVE_INFINITY;

        for (Map.Entry<String, float[][]> entry : classHashes.entrySet()) {
            String label = entry.getKey();
            float[][] hashes = entry.getValue();

            for (float[] hash : hashes) {
                double distance = euclideanDistance(hashVec, hash);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestLabel = label;
                }
            }
        }

        return bestLabel;
    }

    private double euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private Pair<float[][], int[]> kMeans(float[][] data, int maxIterations) {
        int optimalK = findOptimalK(data);
        return kMeans(data, optimalK, maxIterations);
    }

    private int findOptimalK(float[][] data) {
        int minK = 2;
        int maxK = Math.min(10, data.length); // Adjust the upper bound as needed
        double[] inertia = new double[maxK - minK + 1];

        for (int k = minK; k <= maxK; k++) {
            Pair<float[][], int[]> kMeansResult = kMeans(data, k, 300);
            float[][] centroids = kMeansResult.first;
            inertia[k - minK] = calculateInertia(data, centroids);
        }

        int optimalK = elbowMethod(inertia);
        return optimalK + minK;
    }

    private int elbowMethod(double[] inertia) {
        double maxDifference = Double.NEGATIVE_INFINITY;
        int elbow = 0;
        for (int i = 1; i < inertia.length - 1; i++) {
            double difference = Math.abs((inertia[i] - inertia[i - 1]) / (inertia[i + 1] - inertia[i]));
            if (difference > maxDifference) {
                maxDifference = difference;
                elbow = i;
            }
        }
        return elbow;
    }

    private double calculateInertia(float[][] data, float[][] centroids) {
        double inertia = 0;
        for (float[] point : data) {
            double minDistance = Double.POSITIVE_INFINITY;
            for (float[] centroid : centroids) {
                double distance = euclideanDistance(point, centroid);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            inertia += minDistance * minDistance;
        }
        return inertia;
    }

    private float[][] initializeCentroids(float[][] data, int k) {
        float[][] centroids = new float[k][];
        Random random = new Random();
        int firstCentroid = random.nextInt(data.length);
        centroids[0] = data[firstCentroid];

        for (int i = 1; i < k; i++) {
            double[] distances = new double[data.length];
            for (int j = 0; j < data.length; j++) {
                double minDistance = Double.POSITIVE_INFINITY;
                for (int l = 0; l < i; l++) {
                    double distance = euclideanDistance(data[j], centroids[l]);
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
                distances[j] = minDistance;
            }

            double totalDistance = Arrays.stream(distances).sum();
            double randomValue = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0;
            int nextCentroid = -1;
            for (int j = 0; j < data.length; j++) {
                cumulativeDistance += distances[j];
                if (cumulativeDistance >= randomValue) {
                    nextCentroid = j;
                    break;
                }
            }

            centroids[i] = data[nextCentroid];
        }

        return centroids;
    }

    private Pair<float[][], int[]> kMeans(float[][] data, int k, int maxIterations) {
        float[][] centroids = initializeCentroids(data, k);
        int[] assignments = new int[data.length];

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // Assign data points to the nearest centroid
            boolean changed = false;
            for (int i = 0; i < data.length; i++) {
                float[] point = data[i];
                int nearestCentroid = -1;
                double minDistance = Double.POSITIVE_INFINITY;
                for (int j = 0; j < centroids.length; j++) {
                    float[] centroid = centroids[j];
                    double distance = euclideanDistance(point, centroid);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCentroid = j;
                    }
                }
                if (assignments[i] != nearestCentroid) {
                    assignments[i] = nearestCentroid;
                    changed = true;
                }
            }

            // Update centroids
            float[][] newCentroids = new float[k][data[0].length];
            int[] counts = new int[k];

            for (int i = 0; i < data.length; i++) {
                int centroidIndex = assignments[i];
                float[] point = data[i];
                for (int j = 0; j < data[0].length; j++) {
                    newCentroids[centroidIndex][j] += point[j];
                }
                counts[centroidIndex]++;
            }

            for (int i = 0; i < k; i++) {
                if (counts[i] != 0) {
                    for (int j = 0; j < data[0].length; j++) {
                        newCentroids[i][j] /= counts[i];
                    }
                } else {
                    newCentroids[i] = centroids[i];
                }
            }

            centroids = newCentroids;

            if (!changed) {
                break;
            }
        }

        return new Pair<>(centroids, assignments);
    }

    private class CustomPagerAdapter extends PagerAdapter {
        private Context context;
        private ArrayList<JSONObject> products;
        private ArrayList<JSONObject> sortedByPrice;
        private ArrayList<JSONObject> sortedByRating;
        private ArrayList<JSONObject> sortedByCalories;
        private Bitmap[] bitmaps;

        public CustomPagerAdapter(Context context, ArrayList<JSONObject> products, Bitmap[] bitmaps) {
            this.context = context;
            this.products = products;
            this.bitmaps = bitmaps;
            this.sortedByPrice = new ArrayList<>(products);
            this.sortedByRating = new ArrayList<>(products);
            this.sortedByCalories = new ArrayList<>();

            Collections.sort(sortedByPrice, new Comparator<JSONObject>() {
                @Override
                public int compare(JSONObject p1, JSONObject p2) {
                    try {
                        return Double.compare(p1.getDouble("discountPrice"), p2.getDouble("discountPrice"));
                    } catch (JSONException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }
            });

            Collections.sort(sortedByRating, new Comparator<JSONObject>() {
                @Override
                public int compare(JSONObject p1, JSONObject p2) {
                    try {
                        return Double.compare(p2.getDouble("averageRating"), p1.getDouble("averageRating"));
                    } catch (JSONException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }
            });

            for (JSONObject product : products) {
                try {
                    String caloriesStr = product.getString("calories100g");
                    int calories = Integer.parseInt(caloriesStr);
                    product.put("caloriesInt", calories);
                    sortedByCalories.add(product);
                } catch (JSONException | NumberFormatException e) {
                    // Ignore products with non-integer or missing calories
                }
            }

            Collections.sort(sortedByCalories, new Comparator<JSONObject>() {
                @Override
                public int compare(JSONObject p1, JSONObject p2) {
                    try {
                        return Integer.compare(p1.getInt("caloriesInt"), p2.getInt("caloriesInt"));
                    } catch (JSONException e) {
                        e.printStackTrace();
                        return 0;
                    }
                }
            });
        }

        @Override
        public int getCount() {
            return 3;
        }

        @Override
        public boolean isViewFromObject(@NonNull View view, @NonNull Object object) {
            return view == object;
        }

        @NonNull
        @Override
        public Object instantiateItem(@NonNull ViewGroup container, int position) {
            LayoutInflater inflater = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            View view = inflater.inflate(R.layout.product_list, container, false);

            TableLayout tableLayout = view.findViewById(R.id.table_layout);
            TextView tableTitle = view.findViewById(R.id.table_title);
            tableTitle.setText(position == 0 ? "Price" : position == 1 ? "Rating" : "Calories");

            ArrayList<JSONObject> currentList = position == 0 ? sortedByPrice : position == 1 ? sortedByRating : sortedByCalories;

            Display display = getWindowManager().getDefaultDisplay();
            int screenWidth = display.getWidth();
            int scaledWidth = screenWidth / 3;
            int rowIndex = 0;
            for (JSONObject product : currentList) {
                try {
                    TableRow tableRow = new TableRow(context);

                    // Apply custom row background colors
                    int bgColor = rowIndex % 2 == 0 ? Color.parseColor("#F0F0F0") : Color.parseColor("#FFFFFF");
                    tableRow.setBackgroundColor(bgColor);

                    // Get the index of the product in the original product list
                    int index = products.indexOf(product);

                    ImageView imageView = new ImageView(context);
                    imageView.setImageBitmap(Bitmap.createScaledBitmap(bitmaps[index], scaledWidth, bitmaps[index].getHeight() * scaledWidth / bitmaps[index].getWidth(), true));
                    imageView.setLayoutParams(new TableRow.LayoutParams(0, TableRow.LayoutParams.WRAP_CONTENT, 1));
                    tableRow.addView(imageView);

                    TextView nameView = new TextView(context);
                    nameView.setText(product.getString("title"));
                    nameView.setPadding(8, 8, 8, 8);
                    nameView.setMaxWidth(200);
                    nameView.setHorizontallyScrolling(false);
                    nameView.setLines(10);
                    nameView.setEllipsize(TextUtils.TruncateAt.END);
                    nameView.setLayoutParams(new TableRow.LayoutParams(0, TableRow.LayoutParams.WRAP_CONTENT, 1));
                    tableRow.addView(nameView);

                    TextView valueView = new TextView(context);
                    if (position == 0) {
                        valueView.setText(String.valueOf(product.getDouble("discountPrice")));
                    } else if (position == 1) {
                        valueView.setText(String.valueOf(product.getDouble("averageRating")));
                    } else {
                        valueView.setText(product.getString("calories100g"));
                    }

                    valueView.setPadding(8, 8, 8, 8);
                    valueView.setMaxWidth(100);
                    valueView.setHorizontallyScrolling(false);
                    valueView.setLines(5);
                    valueView.setEllipsize(TextUtils.TruncateAt.END);
                    valueView.setLayoutParams(new TableRow.LayoutParams(0, TableRow.LayoutParams.WRAP_CONTENT, 1));
                    tableRow.addView(valueView);

                    tableLayout.addView(tableRow);

                    rowIndex++;
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }

            container.addView(view);
            return view;
        }

        @Override
        public void destroyItem(@NonNull ViewGroup container, int position, @NonNull Object object) {
            container.removeView((View) object);
        }
    }
}
