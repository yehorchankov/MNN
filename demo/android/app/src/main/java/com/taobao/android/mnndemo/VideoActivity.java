package com.taobao.android.mnndemo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Typeface;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewStub;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.CompoundButton;
import android.widget.FrameLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.ShiftList;
import com.taobao.android.utils.Statistics;
import com.taobao.android.utils.TxtFileReader;

import java.text.DecimalFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class VideoActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private final String TAG = "VideoActivity";
    private final int MAX_CLZ_SIZE = 1000;

    private final String MobileModelFileName = "MobileNet/v2/mobilenet_v2.caffe.mnn";
    private final String MobileWordsFileName = "MobileNet/synset_words.txt";

    private final String SqueezeModelFileName = "SqueezeNet/v1.1/squeezenet_v1.1.caffe.mnn";
    private final String SqueezeWordsFileName = "SqueezeNet/squeezenet.txt";

    private final String UltraNetModelFileName = "UltraNet/slim-320.mnn";

    private final String ShuffleNetModelFileName = "ShuffleNet/shufflenet_64_simple.mnn";

    private String mMobileModelPath;
    private List<String> mMobileTaiWords;
    private String mSqueezeModelPath;
    private List<String> mSqueezeTaiWords;
    private String mUltraModelPath;
    private String mShuffleModelPath;

    private int mSelectedModelIndex;// current using modle
    private final MNNNetInstance.Config mConfig = new MNNNetInstance.Config();// session config

    private CameraView mCameraView;
    private SurfaceView mFaceView;
    private SurfaceHolder mFaceSurfaceHolder;
    private Spinner mForwardTypeSpinner;
    private Spinner mThreadNumSpinner;
    private Spinner mModelSpinner;
    private Spinner mMoreDemoSpinner;

    private TextView mFirstResult;
    private TextView mSecondResult;
    private TextView mThirdResult;
    private TextView mTimeTextView;

    private ShiftList<Double> totalTime;
    private ShiftList<Double> inferenceTime;
    private ShiftList<Double> postprocessTime;
    private final int maxListCapacity = 30;

    private final int MobileInputWidth = 224;
    private final int MobileInputHeight = 224;

    private final int SqueezeInputWidth = 227;
    private final int SqueezeInputHeight = 227;

    private final int UltraInputWidth = 320;
    private final int UltraInputHeight = 240;

    private final int ShuffleInputWidth = 64;
    private final int ShuffleInputHeight = 64;

    private long mFd = 0;

    HandlerThread mThread;
    Handler mHandle;

    private AtomicBoolean mLockUIRender = new AtomicBoolean(false);
    private AtomicBoolean mDrop = new AtomicBoolean(false);

    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;

    private int mRotateDegree;// 0/90/180/360

    /**
     * 监听屏幕旋转
     */
    void detectScreenRotate() {
        OrientationEventListener orientationListener = new OrientationEventListener(this,
                SensorManager.SENSOR_DELAY_NORMAL) {
            @Override
            public void onOrientationChanged(int orientation) {

                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) {
                    return;  //手机平放时，检测不到有效的角度
                }

                //可以根据不同角度检测处理，这里只检测四个角度的改变
                orientation = (orientation + 45) / 90 * 90;
                mRotateDegree = orientation % 360;
            }
        };


        if (orientationListener.canDetectOrientation()) {
            orientationListener.enable();
        } else {
            orientationListener.disable();
        }
    }

    private void prepareModels() {

        mMobileModelPath = getCacheDir() + "mobilenet_v1.caffe.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), MobileModelFileName, mMobileModelPath);
            mMobileTaiWords = TxtFileReader.getUniqueUrls(getBaseContext(), MobileWordsFileName, Integer.MAX_VALUE);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        mSqueezeModelPath = getCacheDir() + "squeezenet_v1.1.caffe.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), SqueezeModelFileName, mSqueezeModelPath);
            mSqueezeTaiWords = TxtFileReader.getUniqueUrls(getBaseContext(), SqueezeWordsFileName, Integer.MAX_VALUE);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        mUltraModelPath = getCacheDir() + "slim-320.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), UltraNetModelFileName, mUltraModelPath);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        mShuffleModelPath = getCacheDir() + "shufflenet_64_simple.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), ShuffleNetModelFileName, mShuffleModelPath);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }


    private void prepareNet() {
        if (null != mSession) {
            mSession.release();
            mSession = null;
        }
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }

        String modelPath = mMobileModelPath;
        if (mSelectedModelIndex == 0) {
            modelPath = mMobileModelPath;
        } else if (mSelectedModelIndex == 1) {
            modelPath = mSqueezeModelPath;
        } else if (mSelectedModelIndex == 2) {
            modelPath = mUltraModelPath;
        } else if (mSelectedModelIndex == 3) {
            modelPath = mShuffleModelPath;
        }

        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(modelPath);

        // mConfig.saveTensors;
        mSession = mNetInstance.createSession(mConfig);

        // get input tensor
        mInputTensor = mSession.getInput(null);

        int[] dimensions = mInputTensor.getDimensions();
        dimensions[0] = 1; // force batch = 1  NCHW  [batch, channels, height, width]
        mInputTensor.reshape(dimensions);
        mSession.reshape();

        mFd = mSession.initFd(320, 240, 3);

        mLockUIRender.set(false);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        detectScreenRotate();

        mSelectedModelIndex = 0;
        mConfig.numThread = 4;
        mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;

        // prepare mnn net models
        prepareModels();

        mForwardTypeSpinner = (Spinner) findViewById(R.id.forwardTypeSpinner);
        mThreadNumSpinner = (Spinner) findViewById(R.id.threadsSpinner);
        mThreadNumSpinner.setSelection(2);
        mModelSpinner = (Spinner) findViewById(R.id.modelTypeSpinner);
        mMoreDemoSpinner = (Spinner) findViewById(R.id.MoreDemo);

        mFirstResult = findViewById(R.id.firstResult);
        mSecondResult = findViewById(R.id.secondResult);
        mThirdResult = findViewById(R.id.thirdResult);
        mTimeTextView = findViewById(R.id.timeTextView);

        mForwardTypeSpinner.setOnItemSelectedListener(VideoActivity.this);
        mThreadNumSpinner.setOnItemSelectedListener(VideoActivity.this);
        mModelSpinner.setOnItemSelectedListener(VideoActivity.this);
        mMoreDemoSpinner.setOnItemSelectedListener(VideoActivity.this);

        // init sub thread handle
        mLockUIRender.set(true);
        clearUIForPrepareNet();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 10);
            } else {
                handlePreViewCallBack();
            }
        } else {
            handlePreViewCallBack();
        }

        mThread = new HandlerThread("MNNNet");
        mThread.start();
        mHandle = new Handler(mThread.getLooper());

        mHandle.post(new Runnable() {
            @Override
            public void run() {
                prepareNet();
            }
        });

    }


    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (10 == requestCode) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                handlePreViewCallBack();
            } else {
                Toast.makeText(this, "没有获得必要的权限", Toast.LENGTH_SHORT).show();
            }
        }

    }

    private void handlePreViewCallBack() {

        ViewStub stub = (ViewStub) findViewById(R.id.stub);
        stub.inflate();

        mCameraView = (CameraView) findViewById(R.id.camera_view);
        mFaceView = (SurfaceView) findViewById(R.id.faceView);
        mFaceSurfaceHolder = mFaceView.getHolder();
        mFaceView.setZOrderOnTop(false);
        mFaceSurfaceHolder.setFormat(PixelFormat.TRANSPARENT);

        Switch camSwitch = (Switch) findViewById(R.id.camSwitch);
        camSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                mCameraView.switchCamera();
            }
        });


        mCameraView.setPreviewCallback(new CameraView.PreviewCallback() {
            @Override
            public void onGetPreviewOptimalSize(int optimalWidth, int optimalHeight) {

                // adjust video preview size according to screen size
                DisplayMetrics metric = new DisplayMetrics();
                getWindowManager().getDefaultDisplay().getMetrics(metric);
                int fixedVideoHeight = metric.widthPixels * optimalWidth / optimalHeight;

                FrameLayout layoutVideo = findViewById(R.id.videoLayout);
                RelativeLayout.LayoutParams params = (RelativeLayout.LayoutParams) layoutVideo.getLayoutParams();
                params.height = fixedVideoHeight;
                layoutVideo.setLayoutParams(params);
            }

            @Override
            public void onPreviewFrame(final byte[] data, final int imageWidth, final int imageHeight, final int angle) {

                if (mLockUIRender.get()) {
                    return;
                }

                if (mDrop.get()) {
                    Log.w(TAG, "drop frame , net running too slow !!");
                } else {
                    mDrop.set(true);
                    mHandle.post(new Runnable() {
                        @Override
                        public void run() {
                            mDrop.set(false);
                            if (mLockUIRender.get()) {
                                return;
                            }

                            // calculate corrected angle based on camera orientation and mobile rotate degree. (back camrea)
                            int needRotateAngle = (angle + mRotateDegree) % 360;

                            /*
                             *  convert data to input tensor
                             */
                            final MNNImageProcess.Config config = new MNNImageProcess.Config();

                            // Handle UltraNet here
                            if (mSelectedModelIndex == 2) {
                                // normalization params
                                config.mean = new float[] {127f, 127f, 127f};
                                config.normal = new float[] {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
                                config.source = MNNImageProcess.Format.YUV_NV21;// input source format
                                config.dest = MNNImageProcess.Format.BGR;// input data format
                                config.wrap = MNNImageProcess.Wrap.REPEAT;
                                config.filter = MNNImageProcess.Filter.BILINEAL;

                                // matrix transform: dst to src
                                Matrix matrix = new Matrix();
                                matrix.setScale(1.0f / imageWidth, 1.0f / imageHeight);
                                matrix.postRotate(needRotateAngle, 0.5f, 0.5f);
                                matrix.postScale(UltraInputWidth, UltraInputHeight);
                                matrix.invert(matrix);

                                MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);

                                final long startTimestamp = System.nanoTime();
                                mSession.run();
                                final long endInferenceTimestamp = System.nanoTime();

                                MNNNetInstance.Session.Tensor scoresOutput = mSession.getOutput("scores");
                                float[] scoresResult = scoresOutput.getFloatData();// get float results

                                MNNNetInstance.Session.Tensor boxesOutput = mSession.getOutput("boxes");
                                float[] boxesResult = boxesOutput.getFloatData();// get float results

                                final float[] faces = mSession.detectFaces(mFd, scoresResult, boxesResult);
                                final long endPostprocessingTimestamp = System.nanoTime();

                                totalTime.add((endPostprocessingTimestamp - startTimestamp) / 1000000.0);
                                inferenceTime.add((endInferenceTimestamp - startTimestamp) / 1000000.0);
                                postprocessTime.add((endPostprocessingTimestamp - endInferenceTimestamp) / 1000000.0);

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        mTimeTextView.setText("");
                                        mFirstResult.setText(String.format("Total time: %.5f ms", Statistics.mean(totalTime)));
                                        mSecondResult.setText(String.format("Inference: %.5f ms", Statistics.mean(inferenceTime)));
                                        mThirdResult.setText(String.format("Postprocessing: %.5f ms", Statistics.mean(postprocessTime)));

                                        Canvas canvas = mFaceSurfaceHolder.lockCanvas();

                                        if (canvas == null) {
                                            return;
                                        }

                                        int canvasWidth = canvas.getWidth();
                                        int canvasHeight = canvas.getHeight();

                                        canvas.drawColor( 0, PorterDuff.Mode.CLEAR );

                                        Paint rectPaint = new Paint();
                                        rectPaint.setColor(Color.CYAN);
                                        rectPaint.setStyle(Paint.Style.STROKE);
                                        rectPaint.setStrokeWidth(5);

                                        Paint textPaint = new Paint();
                                        textPaint.setColor(Color.CYAN);
                                        textPaint.setTextSize(30);

                                        for (int i = 0; i < faces.length / 5; i++) {
                                            // Invert x for front camera image
                                            float x1 = mCameraView.isFrontCamera() ? canvasWidth - faces[i*5] / UltraInputWidth * canvasWidth : faces[i*5] / UltraInputWidth * canvasWidth;
                                            float y1 = faces[i*5 + 1] / UltraInputHeight * canvasHeight;
                                            // Invert x for front camera image
                                            float x2 = mCameraView.isFrontCamera() ? canvasWidth - faces[i*5 + 2] / UltraInputWidth * canvasWidth : faces[i*5 + 2] / UltraInputWidth * canvasWidth;
                                            float y2 = faces[i*5 + 3] / UltraInputHeight * canvasHeight;
                                            float scores = faces[i*5 + 4];

                                            try {
                                                canvas.drawRect(x1, y1, x2, y2, rectPaint);
                                                canvas.drawText(String.format("%.3f", scores), mCameraView.isFrontCamera() ? x2 : x1, y1, textPaint);
                                            } catch (Throwable t) {
                                                Log.e(Common.TAG, "Draw result error:" + t);
                                            }
                                        }

                                        mFaceSurfaceHolder.unlockCanvasAndPost(canvas);
                                    }
                                });

                                return;

                            } else if (mSelectedModelIndex == 0) {
                                // normalization params
                                config.mean = new float[]{103.94f, 116.78f, 123.68f};
                                config.normal = new float[]{0.017f, 0.017f, 0.017f};
                                config.source = MNNImageProcess.Format.YUV_NV21;// input source format
                                config.dest = MNNImageProcess.Format.BGR;// input data format

                                // matrix transform: dst to src
                                Matrix matrix = new Matrix();
                                matrix.postScale(MobileInputWidth / (float) imageWidth, MobileInputHeight / (float) imageHeight);
                                matrix.postRotate(needRotateAngle, MobileInputWidth / 2, MobileInputHeight / 2);
                                matrix.invert(matrix);

                                MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);

                            } else if (mSelectedModelIndex == 1) {
                                // input data format
                                config.source = MNNImageProcess.Format.YUV_NV21;// input source format
                                config.dest = MNNImageProcess.Format.BGR;// input data format

                                // matrix transform: dst to src
                                final Matrix matrix = new Matrix();
                                matrix.postScale(SqueezeInputWidth / (float) imageWidth, SqueezeInputHeight / (float) imageHeight);
                                matrix.postRotate(needRotateAngle, SqueezeInputWidth / 2, SqueezeInputWidth / 2);
                                matrix.invert(matrix);

                                MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);
                            } else if (mSelectedModelIndex == 3) {
                                // input data format
                                config.mean = new float[]{127f, 127f, 127f};
                                config.normal = new float[]{1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
                                config.source = MNNImageProcess.Format.YUV_NV21;// input source format
                                config.dest = MNNImageProcess.Format.BGR;// input data format
                                config.wrap = MNNImageProcess.Wrap.REPEAT;
                                config.filter = MNNImageProcess.Filter.BILINEAL;

                                // matrix transform: dst to src
                                final Matrix matrix = new Matrix();
                                matrix.postScale(ShuffleInputWidth / (float) imageWidth, ShuffleInputHeight / (float) imageHeight);
                                matrix.postRotate(needRotateAngle, ShuffleInputWidth / 2, ShuffleInputWidth / 2);
                                matrix.invert(matrix);

                                MNNImageProcess.convertBuffer(data, imageWidth, imageHeight, mInputTensor, config, matrix);

                                final long startTimestamp = System.nanoTime();
                                mSession.run();
                                final long endInferenceTimestamp = System.nanoTime();

                                MNNNetInstance.Session.Tensor scoresOutput = mSession.getOutput(null);
                                float[] scoresResult = scoresOutput.getFloatData();// get float results

                                inferenceTime.add((endInferenceTimestamp - startTimestamp) / 1000000.0);

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        mTimeTextView.setText("");
                                        mFirstResult.setText(String.format("Inference: %.5f ms", Statistics.mean(inferenceTime)));
                                        mSecondResult.setText("");
                                        mThirdResult.setText("");
                                    }
                                });

                                return;
                            }
                            // Clear canvas
                            try {
                                Canvas canvas = mFaceSurfaceHolder.lockCanvas();
                                canvas.drawColor( 0, PorterDuff.Mode.CLEAR );
                                mFaceSurfaceHolder.unlockCanvasAndPost(canvas);
                            } catch (Throwable t) {
                                Log.e(Common.TAG, "Clear canvas error:" + t);
                            }

                            final long startTimestamp = System.nanoTime();
                            /**
                             * inference
                             */
                            mSession.run();

                            /**
                             * get output tensor
                             */
                            MNNNetInstance.Session.Tensor output = mSession.getOutput(null);

                            float[] result = output.getFloatData();// get float results
                            final long endTimestamp = System.nanoTime();
                            final float inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f;

                            if (result.length > MAX_CLZ_SIZE) {
                                Log.w(TAG, "session result too big (" + result.length + "), model incorrect ?");
                            }

                            final List<Map.Entry<Integer, Float>> maybes = new ArrayList<>();
                            for (int i = 0; i < result.length; i++) {
                                float confidence = result[i];
                                if (confidence > 0.01) {
                                    maybes.add(new AbstractMap.SimpleEntry<Integer, Float>(i, confidence));
                                }
                            }

                            Collections.sort(maybes, new Comparator<Map.Entry<Integer, Float>>() {
                                @Override
                                public int compare(Map.Entry<Integer, Float> o1, Map.Entry<Integer, Float> o2) {
                                    if (Math.abs(o1.getValue() - o2.getValue()) <= Float.MIN_NORMAL) {
                                        return 0;
                                    }
                                    return o1.getValue() > o2.getValue() ? -1 : 1;
                                }
                            });

                            // show results on ui
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {

                                    if (maybes.size() == 0) {
                                        mFirstResult.setText("no data");
                                        mSecondResult.setText("");
                                        mThirdResult.setText("");
                                    }
                                    if (maybes.size() > 0) {
                                        mFirstResult.setTextColor(maybes.get(0).getValue() > 0.2 ? Color.BLACK : Color.parseColor("#a4a4a4"));
                                        final Integer iKey = maybes.get(0).getKey();
                                        final Float fValue = maybes.get(0).getValue();
                                        String strWord = "unknown";
                                        if (0 == mSelectedModelIndex) {
                                            if (iKey < mMobileTaiWords.size()) {
                                                strWord = mMobileTaiWords.get(iKey);
                                            }
                                        } else {
                                            if (iKey < mSqueezeTaiWords.size()) {
                                                strWord = mSqueezeTaiWords.get(iKey);
                                            }
                                        }
                                        final String resKey = mSelectedModelIndex == 1 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mFirstResult.setText(resKey + "：" + new DecimalFormat("0.00").format(fValue));

                                    }
                                    if (maybes.size() > 1) {
                                        final Integer iKey = maybes.get(1).getKey();
                                        final Float fValue = maybes.get(1).getValue();
                                        String strWord = "unknown";
                                        if (0 == mSelectedModelIndex) {
                                            if (iKey < mMobileTaiWords.size()) {
                                                strWord = mMobileTaiWords.get(iKey);
                                            }
                                        } else {
                                            if (iKey < mSqueezeTaiWords.size()) {
                                                strWord = mSqueezeTaiWords.get(iKey);
                                            }
                                        }
                                        final String resKey = mSelectedModelIndex == 1 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mSecondResult.setText(resKey + "：" + new DecimalFormat("0.00").format(fValue));

                                    }
                                    if (maybes.size() > 2) {
                                        final Integer iKey = maybes.get(2).getKey();
                                        final Float fValue = maybes.get(2).getValue();
                                        String strWord = "unknown";
                                        if (0 == mSelectedModelIndex) {
                                            if (iKey < mMobileTaiWords.size()) {
                                                strWord = mMobileTaiWords.get(iKey);
                                            }
                                        } else {
                                            if (iKey < mSqueezeTaiWords.size()) {
                                                strWord = mSqueezeTaiWords.get(iKey);
                                            }
                                        }
                                        final String resKey = mSelectedModelIndex == 1 ? strWord.length() >= 10 ? strWord.substring(10) : strWord : strWord;
                                        mThirdResult.setText(resKey + "：" + new DecimalFormat("0.00").format(fValue));
                                    }

                                    mTimeTextView.setText("cost time：" + inferenceTimeCost + "ms");
                                }
                            });

                        }
                    });
                }
            }
        });
    }


    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        // forward type
        if (mForwardTypeSpinner.getId() == adapterView.getId()) {

            if (i == 0) {
                mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
            } else if (i == 1) {
                mConfig.forwardType = MNNForwardType.FORWARD_OPENCL.type;
            } else if (i == 2) {
                mConfig.forwardType = MNNForwardType.FORWARD_OPENGL.type;
            } else if (i == 3) {
                mConfig.forwardType = MNNForwardType.FORWARD_VULKAN.type;
            }
        }
        // threads num
        else if (mThreadNumSpinner.getId() == adapterView.getId()) {

            String[] threadList = getResources().getStringArray(R.array.thread_list);
            mConfig.numThread = Integer.parseInt(threadList[i].split(" ")[1]);
        }
        // model index
        else if (mModelSpinner.getId() == adapterView.getId()) {

            mSelectedModelIndex = i;
        } else if (mMoreDemoSpinner.getId() == adapterView.getId()) {

            if (i == 1) {
                Intent intent = new Intent(VideoActivity.this, ImageActivity.class);
                startActivity(intent);
            } else if (i == 2) {
                Intent intent = new Intent(VideoActivity.this, PortraitActivity.class);
                startActivity(intent);
            } else if (i == 3) {
                Intent intent = new Intent(VideoActivity.this, OpenGLTestActivity.class);
                startActivity(intent);
            }
        }

        totalTime = new ShiftList<Double>(maxListCapacity);
        inferenceTime = new ShiftList<Double>(maxListCapacity);
        postprocessTime = new ShiftList<Double>(maxListCapacity);

        mLockUIRender.set(true);
        clearUIForPrepareNet();

        mHandle.post(new Runnable() {
            @Override
            public void run() {
                prepareNet();
            }
        });

    }

    private void clearUIForPrepareNet() {
        mFirstResult.setText("prepare net ...");
        mSecondResult.setText("");
        mThirdResult.setText("");
        mTimeTextView.setText("");
    }


    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }

    @Override
    protected void onPause() {
        mCameraView.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mCameraView.onResume();
    }


    @Override
    protected void onDestroy() {
        mThread.interrupt();

        /**
         * instance release
         */
        mHandle.post(new Runnable() {
            @Override
            public void run() {
                if (mNetInstance != null) {
                    mNetInstance.release();
                }
                if (mFd != 0 && mSession != null) {
                    mFd = mSession.releaseFd(mFd);
                }
            }
        });

        super.onDestroy();
    }
}