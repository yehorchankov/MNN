//
// Created by yehor on 03.03.20.
//

#ifndef ANDROID_ULTRAFACE_HPP
#define ANDROID_ULTRAFACE_HPP

#pragma once

#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

class FaceDetection {
public:
    FaceDetection(int inputWidth, int inputHeight, int inputChannels, int numThreads = 4, float scoreThreshold = 0.7, float iouThreshold = 0.35);

    void generateBboxes(std::vector<FaceInfo> &bbox_collection, std::vector<float> scores,
                        std::vector<float> boxes);

    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int in_c;
    int num_anchors;

    float score_threshold;
    float iou_threshold;

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;
    std::vector<std::vector<float>> priors = {};

};

#endif //ANDROID_ULTRAFACE_HPP