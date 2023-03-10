//
// Created by asesorov on 07.03.2023.
//

#include "squeezenet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

SqueezeNet::SqueezeNet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int SqueezeNet::load(AAssetManager* mgr, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    squeezenet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    squeezenet.opt = ncnn::Option();

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = use_gpu;
#endif

    squeezenet.opt.num_threads = ncnn::get_big_cpu_count();
    squeezenet.opt.blob_allocator = &blob_pool_allocator;
    squeezenet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "squeezenet_v1.1.param.bin");
    sprintf(modelpath, "squeezenet_v1.1.bin");

    squeezenet.load_param_bin(mgr, parampath);
    squeezenet.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    // init words
    AAsset* asset = AAssetManager_open(mgr, "synset_words.txt", AASSET_MODE_BUFFER);
    int len = AAsset_getLength(asset);
    std::string words_buffer;
    words_buffer.resize(len);
    int ret = AAsset_read(asset, (void*)words_buffer.data(), len);

    AAsset_close(asset);
    squeezenet_words = split_string(words_buffer, "\n");

    return 0;
}

std::string SqueezeNet::classify(const cv::Mat& rgb, Object obj, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    std::vector<float> cls_scores;

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.input(squeezenet_v1_1_param_id::BLOB_data, in);

    ncnn::Mat out;
    ex.extract(squeezenet_v1_1_param_id::BLOB_prob, out);

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    // return top class
    int top_class = 0;
    float max_score = 0.f;
    for (size_t i=0; i<cls_scores.size(); i++)
    {
        float s = cls_scores[i];
//         __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%d %f", i, s);
        if (s > max_score)
        {
            top_class = i;
            max_score = s;
        }
    }

    const std::string& word = squeezenet_words[top_class];
    char tmp[32];
    sprintf(tmp, "%.3f", max_score);
    std::string result_str = std::string(word.c_str() + 10) + " = " + tmp;

    // +10 to skip leading n03179701
    jstring result = env->NewStringUTF(result_str.c_str());

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "SqueezeNcnn", "%.2fms   detect", elasped);

    return result;
}
