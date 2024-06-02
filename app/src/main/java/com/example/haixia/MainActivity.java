package com.example.haixia;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;

    // 定义常量
    private Scalar lowerGreen = new Scalar(40, 40, 40);
    private Scalar upperGreen = new Scalar(120, 220, 220);
    private int contourAreaThreshold = 4000;

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        // 判断OpenCV是否加载成功
        if (OpenCVLoader.initLocal()) {
            Log.i("OpenCV", "OpenCV loaded successfully");
        } else {
            Log.e("OpenCV", "OpenCV not loaded");
            (Toast.makeText(this, "OpenCV init failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        // 保持屏幕常亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // 设置全屏模式，隐藏状态栏
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.tutorial1_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Mat hsv = new Mat();
        Mat mask = new Mat();
        Mat result = new Mat();
        Mat gray = new Mat();
        Mat edged = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_RGB2HSV);
        // 通过颜色阈值分割出绿色区域，将其存储在mask中（绿色区域为白色，其余为黑色）
        Core.inRange(hsv, lowerGreen, upperGreen, mask);
        // 将原图像与mask按位与操作，将结果存储在result中
        Core.bitwise_and(frame, frame, result, mask);
        Imgproc.GaussianBlur(result, gray, new Size(7, 7), 0);
        Imgproc.Canny(gray, edged, 50, 100);
        // 闭运算：消除图像的小噪声
        Imgproc.dilate(edged, edged, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(edged, edged, new Mat(), new Point(-1, -1), 1);
        // 轮廓检测，将检测到的轮廓存储在contours中
        Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            // 计算轮廓的面积，如果面积小于阈值，则跳过
            if (Imgproc.contourArea(contour) < contourAreaThreshold) {
                continue;
            }
            // 计算轮廓的最小外接矩形，并将rect的四个顶点存储在boxPoints中
            RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            Point[] boxPoints = new Point[4];
            rect.points(boxPoints);

            // 在原图像(frame)中的每个顶点位置画一个圆
            for (Point point : boxPoints) {
                Imgproc.circle(frame, point, 2, new Scalar(255, 0, 0), -1);
            }
            // 在原图像(frame)中画出最小外接矩形rect的边框
            for (int i = 0; i < 4; i++) {
                Imgproc.line(frame, boxPoints[i], boxPoints[(i+1) % 4], new Scalar(255, 0, 255), 2);
            }

            // 计算外接矩形的长和宽
            double dA = distEuclidean(boxPoints[0], boxPoints[1]);
            double dB = distEuclidean(boxPoints[1], boxPoints[2]);
            double length = Math.max(dA, dB);
            double width = Math.min(dA, dB);

            // 将长和宽输出到控制台
            Log.i(TAG, "Length: " + length + ", Width: " + width);
        }


        return frame;
    }

    // 计算两点之间的欧几里得距离
    private double distEuclidean(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

}