//
// Created by asesorov on 07.03.2023.
//

#ifndef SQUEEZENET_H
#define SQUEEZENET_H

#include <opencv2/core/core.hpp>

// ncnn
#include <net.h>
#include <string>

#include "squeezenet_v1.1.id.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class SqueezeNet
{
public:
    SqueezeNet();

    int load(int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    std::string classify(const cv::Mat& rgb, Object obj, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

private:
    ncnn::Net squeezenet;
    static std::vector<std::string> squeezenet_words;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // SQUEEZENET_H
