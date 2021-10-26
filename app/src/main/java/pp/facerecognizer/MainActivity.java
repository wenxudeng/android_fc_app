/*
* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package pp.facerecognizer;

import android.content.ClipData;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

import androidx.appcompat.app.AlertDialog;

import pp.facerecognizer.env.BorderedText;
import pp.facerecognizer.env.FileUtils;
import pp.facerecognizer.env.ImageUtils;
import pp.facerecognizer.env.Logger;
import pp.facerecognizer.tracking.MultiBoxTracker;

/**
* An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
* objects.
*/
public class MainActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int FACE_SIZE = 112;
    private static final int CROP_SIZE = 300;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier classifier;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private Snackbar initSnackbar;
    private Snackbar trainSnackbar;
    private FloatingActionButton button;

    private boolean initialized = false;
    private boolean training = false;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 帧视图中的容器
        FrameLayout container = findViewById(R.id.container);
        initSnackbar = Snackbar.make(container, "Initializing...", Snackbar.LENGTH_INDEFINITE);
        trainSnackbar = Snackbar.make(container, "Training data...", Snackbar.LENGTH_INDEFINITE);

        // 对话框以及编辑文本视图
        View dialogView = getLayoutInflater().inflate(R.layout.dialog_edittext, null);
        // 编辑文本
        EditText editText = dialogView.findViewById(R.id.edit_text);

        // 对话框
        AlertDialog editDialog = new AlertDialog.Builder(MainActivity.this)
                .setTitle(R.string.enter_name)
                .setView(dialogView)
                .setPositiveButton(getString(R.string.ok), (dialogInterface, i) -> {
                    int idx = classifier.addPerson(editText.getText().toString());
                    performFileSearch(idx - 1);
                })
                .create();

        // 加号按钮
        button = findViewById(R.id.add_button);
        // 按钮监听器
        button.setOnClickListener(view ->
                // AlertDialog是一个对话框
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(getString(R.string.select))
                        .setItems(new String[]{getString(R.string.face_registry),getString(R.string.static_image_face_recognition)},(dialogInterface1, index)->{
                            if (index == 0){
                                new AlertDialog.Builder(MainActivity.this)
                                        .setTitle(getString(R.string.select_name))// Select name
                                        .setItems(classifier.getClassNames(), (dialogInterface, i) -> {
                                            if (i == 0) {
                                                editDialog.show();
                                            } else {
                                                performFileSearch(i - 1);
                                            }
                                        })
                                        .show();
                            }else{
                                performFileSearch(Integer.parseInt(getString(R.string.static_image_request_code)));
                            }
                        })
                        .show());
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        if (!initialized)
            new Thread(this::init).start();

        final float textSizePx =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_SIZE, CROP_SIZE,
                        sensorOrientation, false);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        addCallback(
                canvas -> {
                    if (!isDebug()) {
                        return;
                    }
                    final Bitmap copy = cropCopyBitmap;
                    if (copy == null) {
                        return;
                    }

                    final int backgroundColor = Color.argb(100, 0, 0, 0);
                    canvas.drawColor(backgroundColor);

                    final Matrix matrix = new Matrix();
                    final float scaleFactor = 2;
                    matrix.postScale(scaleFactor, scaleFactor);
                    matrix.postTranslate(
                            canvas.getWidth() - copy.getWidth() * scaleFactor,
                            canvas.getHeight() - copy.getHeight() * scaleFactor);
                    canvas.drawBitmap(copy, matrix, new Paint());

                });
    }

    OverlayView trackingOverlay;

    void init() {
        runOnUiThread(()-> initSnackbar.show());
        File dir = new File(FileUtils.ROOT);

        if (!dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();
        }

        try {
            classifier = Classifier.getInstance(getAssets(), FACE_SIZE, FACE_SIZE);
        } catch (Exception e) {
            LOGGER.e("Exception initializing classifier!", e);
            finish();
        }

        runOnUiThread(()-> initSnackbar.dismiss());
        initialized = true;
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection || !initialized || training) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
//        LOGGER.i("Preparing image " + currTimestamp + " for pp.facerecognizer.detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                () -> {
//                    LOGGER.i("Running pp.facerecognizer.detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    List<Classifier.Recognition> mappedRecognitions =
                            classifier.recognizeImage(croppedBitmap,cropToFrameTransform);

                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                    trackingOverlay.postInvalidate();

                    requestRender();
                    computingDetection = false;
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (!initialized) {
            Snackbar.make(
                    getWindow().getDecorView().findViewById(R.id.container),
                    "Try it again later", Snackbar.LENGTH_SHORT)
                    .show();
            return;
        }

        if (resultCode == RESULT_OK) {
            if(requestCode == Integer.parseInt(getString(R.string.static_image_request_code))){
                recognizeStaticImg(data);
            }else {
                trainSnackbar.show();
                button.setEnabled(false);
                training = true;

                ClipData clipData = data.getClipData();
//            Uri uri = clipData.getItemAt(0).getUri();

                ArrayList<Uri> uris = new ArrayList<>();

                if (clipData == null) {
                    uris.add(data.getData());
                } else {
                    for (int i = 0; i < clipData.getItemCount(); i++)
                        uris.add(clipData.getItemAt(i).getUri());
                }

                new Thread(() -> {
                    try {
                        classifier.updateData(requestCode, getContentResolver(), uris);
                    } catch (Exception e) {
                        LOGGER.e(e, "Exception!");
                    } finally {
                        training = false;
                    }
                    runOnUiThread(() -> {
                        trainSnackbar.dismiss();
                        button.setEnabled(true);
                    });
                }).start();
            }

        }
    }

    /**
     * 打开文件夹获取图片
     * @param requestCode
     */
    public void performFileSearch(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, requestCode);
    }

    private void recognizeStaticImg(Intent data){
        ImageView img = new ImageView(MainActivity.this);
        try {
            Bitmap srcBitmap = classifier.getBitmapFromUri(getContentResolver(), data.getData());
            List<Classifier.Recognition> recognitions = classifier.recognizeStaticImage(srcBitmap, new Matrix());

            srcBitmap = srcBitmap.copy(Config.ARGB_8888, true);
            final Canvas canvas = new Canvas(srcBitmap);
            final Paint textPaint = new Paint();
            textPaint.setColor(Color.MAGENTA);
            textPaint.setTextSize(100.0f);

            final Paint boxPaint = new Paint();
            boxPaint.setColor(Color.GREEN);
            boxPaint.setAlpha(255);
            boxPaint.setStrokeWidth(12.0f);
            boxPaint.setStyle(Paint.Style.STROKE);

            for (final Classifier.Recognition recognition : recognitions) {
                final RectF rect = recognition.getLocation();
                canvas.drawRect(rect, boxPaint);
                canvas.drawText("" + recognition.getTitle(), rect.left, rect.top, textPaint);
            }
            img.setImageBitmap(srcBitmap);
            ViewGroup.LayoutParams ivlp = new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
            img.setLayoutParams(ivlp);
            new AlertDialog.Builder(MainActivity.this)
                    .setTitle("识别结果")
                    .setView(img)
                    .setPositiveButton("确定", null)
                    .show();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void saveImage(Bitmap bitmap)  {
        System.out.println("saveImage");
        /*String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
        OutputStream outStream = null;
        String filename;//声明文件名
        //以保存时间为文件名
        Date date = new Date(System.currentTimeMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMddHHmmss");
        filename =  sdf.format(date);
        File file = new File(extStorageDirectory, filename+".JPEG");//创建文件，第一个参数为路径，第二个参数为文件名
        try {
            outStream = new FileOutputStream(file);//创建输入流
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outStream);
            outStream.close();
            // 这三行可以实现相册更新
            Intent intent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
            Uri uri = Uri.fromFile(file);intent.setData(uri);
            sendBroadcast(intent);
            //这个广播的目的就是更新图库，发了这个广播进入相册就可以找到你保存的图片了
            Toast.makeText(MainActivity.this,"saved",
                    Toast.LENGTH_SHORT).show();
        } catch(Exception e) {
            Toast.makeText(MainActivity.this, "exception:" + e,
                    Toast.LENGTH_SHORT).show();

        }*/
    }
}
