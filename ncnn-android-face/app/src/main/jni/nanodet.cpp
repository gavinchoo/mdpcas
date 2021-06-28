// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"


static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(const ncnn::Mat &cls_pred, const ncnn::Mat &dis_pred, int stride,
                               const ncnn::Mat &in_pad, float prob_threshold,
                               std::vector<Object> &objects) {

    const int num_grid = cls_pred.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h) {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    } else {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = cls_pred.w;
    const int reg_max_1 = dis_pred.w / 4;
    //__android_log_print(ANDROID_LOG_WARN, "ncnn","cls_pred h %d, w %d",cls_pred.h,cls_pred.w);
    //__android_log_print(ANDROID_LOG_WARN, "ncnn","%d,%d,%d,%d",num_grid_x,num_grid_y,num_class,reg_max_1);
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            const int idx = i * num_grid_x + j;

            const float *scores = cls_pred.row(idx);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++) {
                if (scores[k] > score) {
                    label = k;
                    score = scores[k];
                }
            }

            if (score >= prob_threshold) {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void *) dis_pred.row(idx));
                {
                    ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(bbox_pred, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float dis = 0.f;
                    const float *dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++) {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

NanoDet::NanoDet() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(const char *modeltype, int _target_size, const float *_mean_vals,
                  const float *_norm_vals, bool use_gpu) {
    nanodetFace.clear();
    nanodetLandmask.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodetFace.opt = ncnn::Option();
    nanodetLandmask.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodetFace.opt.use_vulkan_compute = use_gpu;
    nanodetLandmask.opt.use_vulkan_compute = use_gpu;

#endif

    nanodetFace.opt.num_threads = ncnn::get_big_cpu_count();
    nanodetFace.opt.blob_allocator = &blob_pool_allocator;
    nanodetFace.opt.workspace_allocator = &workspace_pool_allocator;

    nanodetLandmask.opt.num_threads = ncnn::get_big_cpu_count();
    nanodetLandmask.opt.blob_allocator = &blob_pool_allocator;
    nanodetLandmask.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "nanodet-%s.param", modeltype);
    sprintf(modelpath, "nanodet-%s.bin", modeltype);

    nanodetFace.load_param("nanodet_m_face.param");
    nanodetFace.load_model("nanodet_m_face.bin");

    nanodetLandmask.load_param(parampath);
    nanodetLandmask.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int
NanoDet::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_mean_vals,
              const float *_norm_vals, bool use_gpu) {
    __android_log_print(ANDROID_LOG_WARN, "ncnn", "load %s", modeltype);
    nanodetFace.clear();
    nanodetLandmask.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodetFace.opt = ncnn::Option();
    nanodetLandmask.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodetFace.opt.use_vulkan_compute = use_gpu;
    nanodetLandmask.opt.use_vulkan_compute = use_gpu;

#endif

    nanodetFace.opt.num_threads = ncnn::get_big_cpu_count();
    nanodetFace.opt.blob_allocator = &blob_pool_allocator;
    nanodetFace.opt.workspace_allocator = &workspace_pool_allocator;

    nanodetLandmask.opt.num_threads = ncnn::get_big_cpu_count();
    nanodetLandmask.opt.blob_allocator = &blob_pool_allocator;
    nanodetLandmask.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "pfld-%s.param", modeltype);
    sprintf(modelpath, "pfld-%s.bin", modeltype);
    //__android_log_print(ANDROID_LOG_WARN, "ncnn", "load %s,%s", parampath,modelpath);

    nanodetFace.load_param("nanodet_m_face.param");
    nanodetFace.load_model("nanodet_m_face.bin");

    nanodetLandmask.load_param(mgr, parampath);
    nanodetLandmask.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int NanoDet::detectFace(ncnn::Net &nanodet_face, const cv::Mat &rgb, std::vector<Object> &objects,
                        float target_size, float prob_threshold, float nms_threshold) {
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height,
                                                 w, h);

    // pad to target_size rectangle
    int wpad = 320 - w; //(w + 31) / 32 * 32 - w;
    int hpad = 320 - h; //(h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {1.f / 57.375f, 1.f / 57.12f, 1.f / 58.395f};

    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = nanodet_face.create_extractor();
    //__android_log_print(ANDROID_LOG_WARN, "ncnn","input w:%d,h:%d",in_pad.w,in_pad.h);
    ex.input("input.1", in_pad);

    std::vector<Object> proposals;

    // stride 8
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls_pred_stride_8", cls_pred);
        ex.extract("dis_pred_stride_8", dis_pred);

        std::vector<Object> objects8;
        generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls_pred_stride_16", cls_pred);
        ex.extract("dis_pred_stride_16", dis_pred);

        std::vector<Object> objects16;
        generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls_pred_stride_32", cls_pred);
        ex.extract("dis_pred_stride_32", dis_pred);

        std::vector<Object> objects32;
        generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);

    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = (x1 - x0);
        objects[i].rect.height = (y1 - y0);
    }

    // sort objects by area
    struct {
        bool operator()(const Object &a, const Object &b) const {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    __android_log_print(ANDROID_LOG_WARN, "face count ", "load %d", objects.size());

    return 0;
}

int NanoDet::detectLandmask(ncnn::Net &pfld, const cv::Mat &img, Object &face) {
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols,
                                                 img.rows, 112, 112);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = pfld.create_extractor();
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("output", out);

    cv::Mat rgb = img.clone();

    for (int c = 0; c < out.c; c++) {
        ncnn::Mat data = out.channel(c);
        const float *ptr = data.row(0);
        for (size_t j = 0; j < 98; j++) {
            float pt_x = ptr[j * 2] * img.cols;
            float pt_y = ptr[j * 2 + 1] * img.rows;
            face.pts.push_back(cv::Point2f(pt_x + face.rect.x, pt_y + face.rect.y));
        }
    }

    return 0;
}

int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& faces) {
    detectFace(nanodetFace, rgb, faces, 320, 0.1f, 0.1f);
    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Mat roi_image = rgb(faces[i].rect);
        detectLandmask(nanodetLandmask, roi_image, faces[i]);
    }
    return 0;
}


int NanoDet::draw(cv::Mat&rgb, const std::vector<Object>& faces) {
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar cc(255, 0, 0);
        cv::rectangle(rgb, faces[i].rect, cc, 2);
        for (size_t j = 0; j < faces[i].pts.size(); j++) {
            cv::circle(rgb, faces[i].pts[j], 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    return 0;
}
