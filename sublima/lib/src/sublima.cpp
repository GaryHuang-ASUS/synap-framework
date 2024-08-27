/*
 * NDA AND NEED-TO-KNOW REQUIRED
 *
 * Copyright (C) 2013-2020 Synaptics Incorporated. All rights reserved.
 *
 * This file contains information that is proprietary to Synaptics
 * Incorporated ("Synaptics"). The holder of this file shall treat all
 * information contained herein as confidential, shall use the
 * information only for its intended purpose, and shall not duplicate,
 * disclose, or disseminate any of this information in any manner
 * unless Synaptics has otherwise provided express, written
 * permission.
 *
 * Use of the materials may require a license of intellectual property
 * from a third party or from Synaptics. This file conveys no express
 * or implied licenses to any intellectual property rights belonging
 * to Synaptics.
 *
 * INFORMATION CONTAINED IN THIS DOCUMENT IS PROVIDED "AS-IS", AND
 * SYNAPTICS EXPRESSLY DISCLAIMS ALL EXPRESS AND IMPLIED WARRANTIES,
 * INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE, AND ANY WARRANTIES OF NON-INFRINGEMENT OF ANY
 * INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL SYNAPTICS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE, OR
 * CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OF THE INFORMATION CONTAINED IN THIS DOCUMENT, HOWEVER CAUSED AND
 * BASED ON ANY THEORY OF LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, AND EVEN IF SYNAPTICS WAS
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. IF A TRIBUNAL OF
 * COMPETENT JURISDICTION DOES NOT PERMIT THE DISCLAIMER OF DIRECT
 * DAMAGES OR ANY OTHER DAMAGES, SYNAPTICS' TOTAL CUMULATIVE LIABILITY
 * TO ANY PARTY SHALL NOT EXCEED ONE HUNDRED U.S. DOLLARS.
 */

///
/// Sublima HDR image processing
///

#include "synap/network.hpp"
#include "synap/timer.hpp"
#include "synap/types.hpp"
#include "synap/logging.hpp"
#include "synap/file_utils.hpp"
#include "synap/sublima.hpp"

#include <fstream>
#include <future>
#include "json.hpp"
#include <algorithm>

using namespace std;
using json = nlohmann::ordered_json;
static constexpr int json_dump_indent = 2;

namespace synaptics {
namespace synap {

static void from_json(const json& j, Sublima::JsonHDRInfo& p)
{
    if (j.contains("type"))
        j.at("type").get_to(p.type);

    if (j.contains("PrimaryChromaticityR_X"))
        j.at("PrimaryChromaticityR_X").get_to(p.PrimaryChromaticityR_X);
    if (j.contains("PrimaryChromaticityR_Y"))
        j.at("PrimaryChromaticityR_Y").get_to(p.PrimaryChromaticityR_Y);

    if (j.contains("PrimaryChromaticityG_X"))
        j.at("PrimaryChromaticityG_X").get_to(p.PrimaryChromaticityG_X);
    if (j.contains("PrimaryChromaticityG_Y"))
        j.at("PrimaryChromaticityG_Y").get_to(p.PrimaryChromaticityG_Y);

    if (j.contains("PrimaryChromaticityB_X"))
        j.at("PrimaryChromaticityB_X").get_to(p.PrimaryChromaticityB_X);
    if (j.contains("PrimaryChromaticityB_Y"))
        j.at("PrimaryChromaticityB_Y").get_to(p.PrimaryChromaticityB_Y);

    if (j.contains("WhitePointChromaticity_X"))
        j.at("WhitePointChromaticity_X").get_to(p.WhitePointChromaticity_X);
    if (j.contains("WhitePointChromaticity_Y"))
        j.at("WhitePointChromaticity_Y").get_to(p.WhitePointChromaticity_Y);

    if (j.contains("LuminanceMax"))
        j.at("LuminanceMax").get_to(p.LuminanceMax);
    if (j.contains("LuminanceMin"))
        j.at("LuminanceMin").get_to(p.LuminanceMin);

    if (j.contains("MatrixCoefficients"))
        j.at("MatrixCoefficients").get_to(p.MatrixCoefficients);
    if (j.contains("TransferCharacteristics"))
        j.at("TransferCharacteristics").get_to(p.TransferCharacteristics);
    if (j.contains("ColorPrimaries"))
        j.at("ColorPrimaries").get_to(p.ColorPrimaries);

    if (j.contains("MaxCLL"))
        j.at("MaxCLL").get_to(p.MaxCLL);
    if (j.contains("MaxFALL"))
        j.at("MaxFALL").get_to(p.MaxFALL);

}


struct Sublima::Private {
    // Read LUT2D from a CSV file
    static std::vector<uint16_t> load_lut(std::string file_path);

    // Get the LUT value for the given y and median
    inline uint16_t lut(uint8_t y, uint8_t median) const;

    // Convert NCHW 16bits data to packed 10bits NV15 data
    void convert_nchw_nv15(const uint16_t *data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad);

    // Convert 8bits Y plane to 16bits by applying LUT, then pack to 10bits NV15 data
    void convert_lut_nv15(const uint8_t *data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad, uint8_t median);
    
    // Convert 8bits data to packed 10bits NV15 data by inserting 2 LSB at 0 to each pixel
    void convert_nv15(const uint8_t* data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad);
    
    // Assign the input tensor with the Y, UV planes of the NV12 image and the previous median value
    // returns the median of the Y plane
    uint8_t assign_tensor(const uint8_t* nv12y, const uint8_t* nv12uv, uint16_t* data, uint32_t width, uint32_t height);

    //
    bool parse_hdrjson(const std::string& hdrjson);
    
    bool _hdr{true};
    Network _network;
    std::vector<uint16_t> _lut_2d{};
    static constexpr int k_lum_values = 256;
    std::vector<uint32_t> _histogram = std::vector<uint32_t>(k_lum_values);
    Sublima::Timings _timings{};
    bool _only_y{false};
    Sublima::JsonHDRInfo _hdrinfo{};
};

// Look-up table used to clip UV values
static uint8_t s_clip_table[256];

Sublima::Sublima(): d{new Private}
{
    // Init clip table
    for (int i = 0; i < sizeof(s_clip_table) / sizeof(s_clip_table[0]); i++) {
        s_clip_table[i] = std::clamp(i, 16, 240);
    }
}


Sublima::~Sublima()
{
}


bool Sublima::init(const std::string& lut2d,
                   const std::string& model, const std::string& meta,
                   const std::string& hdrjson) {
    if (lut2d == "-") {
        LOGI << "Sublima starting in SDR-only mode";
        d->_hdr = false;
        return true;
    }

    if (!d->_network.load_model(model, meta)) {
        LOGE << "Sublima failed to load network: " << model;
        return false;
    }

    // Initialize the input tensor since the mean plane is updated only when it changes
    memset(d->_network.inputs[0].data(), 0, d->_network.inputs[0].size());

    d->_lut_2d = d->load_lut(lut2d);
    if (d->_lut_2d.empty()) {
        LOGE << "Sublima failed to load LUT2D: " << lut2d;
        return false;
    }

    if (!hdrjson.empty()) {
        d->parse_hdrjson(hdrjson);
    }

    return true;
}

Sublima::JsonHDRInfo Sublima::hdrinfo()
{
    return d->_hdrinfo;
}


bool Sublima::set_hdr(bool enable)
{
    if (enable && d->_lut_2d.empty()) {
        LOGE << "Sublima LUT2D file not loaded, HDR mode not supported";
        return false;
    }
    d->_hdr = enable;
    LOGI << "Sublima HDR: " << enable;
    return true;
}

bool Sublima::set_only_y(bool enable)
{
    d->_only_y = enable;
    return true;
}

bool Sublima::Private::parse_hdrjson(const std::string& hdrjson)
{
    if (!file_exists(hdrjson)) {
        LOGE << "HDR Json Info file not found: " << hdrjson;
        return false;
    }

    std::string info_string = file_read(hdrjson);
    if (info_string.empty()) {
        LOGE << "Failed to read info file: " << hdrjson;
        return false;
    }

    try {
        json j = json::parse(info_string);
        j.get_to(_hdrinfo);
    } catch (json::parse_error& e) {
        LOGE << hdrjson << ": " << e.what();
        return false;
    }

    return true;
}


vector<uint16_t> Sublima::Private::load_lut(string file_path)
{
    if (!file_exists(file_path)) {
        LOGE << "Sublima LUT2D file not found: " << file_path;
        return {};
    }

    ifstream myfile;
    myfile.open(file_path);

    vector<uint16_t> lut(k_lum_values * k_lum_values);
    std::string line;

    // Read lut from file in CSV format.
    // The first line must contain the min and max indexes of the LUT2D
    int lmin, lmax;
    char sep;
    std::getline(myfile, line);
    std::stringstream head(line);
    head >> lmin >> sep >> lmax;
    LOGV << "Sublima reading LUT2D file: " << file_path << " lmin:" << lmin << " lmax:" << lmax;

    if (sep != ',' || lmin > 50 || lmin < 0 || lmax < 200 || lmax >= k_lum_values || lmin >= lmax) {
        LOGE << "Sublima invalid LUT2D range";
        return {};
    }

    // The remaining file must contain (lmax-lmin+1) lines and columns
    for(int r = lmin; r <= lmax && std::getline(myfile, line); r++) {
        std::stringstream ss(line);
        string val;
        for(int c = lmin; c <= lmax && std::getline(ss, val, ','); c++) {
            lut.at(r * k_lum_values + c) = stoi(val);
        }
    }

    return lut;
}


uint16_t Sublima::Private::lut(uint8_t y, uint8_t median) const {
    return _lut_2d[y * k_lum_values + median];
}


inline uint16_t uv_clip(uint8_t val)
{
    // Use a look-up table to clip UV values. This is faster than using std::clamp on each value
    // and much faster than using NEON
    return s_clip_table[val];
}


uint8_t Sublima::Private::assign_tensor(const uint8_t* nv12y, const uint8_t* nv12uv, uint16_t* data, uint32_t width, uint32_t height)
{
    Timer t;
    const uint32_t total_pixels = width * height;
    const uint32_t quarter_pixels = total_pixels / 4;
    
    // Assign the Y plane in quarter resolution and update the histograms for the median
    fill(_histogram.begin(), _histogram.end(), 0);
    const auto histogram = _histogram.data();
    uint16_t* y16 = &data[quarter_pixels];
    for (uint32_t h = 0 ; h < height; h += 2) {
        const uint8_t* y = &nv12y[h * width];
        for (const uint8_t* const end = &y[width]; y < end; y += 2) {
            *y16++ = *y;
            ++histogram[*y];
        }
    }
    
    // Compute the median Y value
    int median = 0;
    for (uint32_t nb_pixels = 0; nb_pixels < quarter_pixels / 2; nb_pixels += histogram[median++]) {
    }
    --median;
    if (data[0] != median) {
        // Median value changed, update input tensor
        fill_n(&data[0], quarter_pixels, median);
    }
    
    // Assign the UV planes
    uint16_t* u16 = &data[quarter_pixels * 2];
    uint16_t* v16 = &data[quarter_pixels * 3];
    const uint8_t* uv = nv12uv;
    for (const uint8_t* const end = &uv[quarter_pixels * 2]; uv < end; ) {
        // Unrolling the loop a bit improves performances considerably
        *u16++ = uv_clip(*uv++);
        *v16++ = uv_clip(*uv++);
        *u16++ = uv_clip(*uv++);
        *v16++ = uv_clip(*uv++);
    }
    _timings.assign += t.get();
    return median;
}

inline void to_nv15(uint16_t data0, uint16_t data1, uint16_t data2, uint16_t data3, uint8_t*& nv15)
{
    *nv15++ = data0 & 0xff;
    *nv15++ = ((data0 >> 8) & 0x03) + ((data1 & 0x3f) << 2);
    *nv15++ = ((data1 >> 6) & 0x0f) + ((data2 & 0x0f) << 4);
    *nv15++ = ((data2 >> 4) & 0x3f) + ((data3 & 0x03) << 6);
    *nv15++ = (data3 >> 2) & 0xff;
}

void Sublima::Private::convert_nv15(const uint8_t* data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad)
{
    const uint32_t count = width * height;
    const uint8_t* d = &data[0];
    const uint8_t* end = &data[count];
    while (d < end) {
        for (uint32_t x = 0; x < width; x += 4) {
            const uint16_t data0 = *d++ << 2;
            const uint16_t data1 = *d++ << 2;
            const uint16_t data2 = *d++ << 2;
            const uint16_t data3 = *d++ << 2;
            to_nv15(data0, data1, data2, data3, nv15);
        }
        nv15 += nv15_pad;
    }
}



void Sublima::Private::convert_nchw_nv15(const uint16_t* data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad)
{
    Timer t;
    const uint32_t count = width * height;
    const uint16_t* u = &data[0];
    const uint16_t* v = &data[count / 2];
    const uint16_t* end = &data[count];
    while (v < end) {
        for (uint32_t x = 0; x < width; x += 4) {
            const uint16_t data0 = *u++;
            const uint16_t data1 = *v++;
            const uint16_t data2 = *u++;
            const uint16_t data3 = *v++;
            to_nv15(data0, data1, data2, data3, nv15);
        }
        nv15 += nv15_pad;
    }

    _timings.nv15uv += t.get();
}


void Sublima::Private::convert_lut_nv15(const uint8_t* data, uint8_t* nv15, uint32_t width, uint32_t height, uint32_t nv15_pad, uint8_t median)
{
    Timer t;
    const uint32_t count = width * height;
    const uint8_t* end = &data[count];
    while (data < end) {
        for (uint32_t x = 0; x < width; x += 4) {
            const uint16_t data0 = lut(*data++, median);
            const uint16_t data1 = lut(*data++, median);
            const uint16_t data2 = lut(*data++, median);
            const uint16_t data3 = lut(*data++, median);
            to_nv15(data0, data1, data2, data3, nv15);
        }
        nv15 += nv15_pad;
    }
    _timings.nv15y += t.get();
}


bool Sublima::process(const uint8_t* nv12y, const uint8_t* nv12uv, uint8_t* nv15y, uint8_t* nv15uv,
                      uint32_t width, uint32_t height, uint32_t nv15_pad) {
    LOGV << "Sublima processing with nv15_pad: " << nv15_pad;
    Timer tot;
    const uint32_t total_pixels = width * height;
    if (!d->_hdr) {
        LOGV << "Sublima HDR not emabled, converting to NV15 only";
        d->convert_nv15(nv12y, nv15y, width, height, nv15_pad);
        d->convert_nv15(nv12uv, nv15uv, width, height / 2, nv15_pad);

        d->_timings.tot += tot.get();
        d->_timings.cnt += 1;
        return true;
    }

    if (d->_network.inputs[0].size() != total_pixels * sizeof(uint16_t)) {
        LOGE << "Sublima tensor size mismatch, expected:" << total_pixels * sizeof(uint16_t)
             << ", got:" << d->_network.inputs[0].size();
        return false;
    }

    // Compute and assign the model input tensor
    auto median = d->assign_tensor(nv12y, nv12uv, static_cast<uint16_t*>(
                                   d->_network.inputs[0].data()), width, height);

    // Perform inference in background
    future<bool> success = async(launch::async, [this]{
        Timer t;
        bool success = d->_network.predict();
        d->_timings.infer += t.get();
        return success;
    });

    // In the meantime, apply the LUT to the Y plane and convert to NV15
    d->convert_lut_nv15(nv12y, nv15y, width, height, nv15_pad, median);

    // Wait for inference to complete, then convert the UV planes to NV15
    if (!success.get()) {
        LOGE << "Sublima inference failed";
        return false;
    }

    if (!d->_only_y) {
        d->convert_nchw_nv15(static_cast<uint16_t*>(d->_network.outputs[0].data()), nv15uv, width, height / 2, nv15_pad);
    }

    d->_timings.tot += tot.get();
    d->_timings.cnt += 1;
    return true;
}


Sublima::Timings* Sublima::timings()
{
    return &d->_timings;
}

}  // namespace synap
}  // namespace synaptics
