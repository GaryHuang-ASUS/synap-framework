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
#include "synap/sr_sublima.hpp"

#include <iostream>
#include <fstream>
#include <future>

using namespace std;

namespace synaptics {
namespace synap {

struct SrSublima::Private {

    Sublima _sublima;
    Network _srnetwork;
    SrSublima::Timings _superResTimings{};
    void dump_sr_output_nv12(const std::string& output_filename_raw, const synaptics::synap::Tensor* y_tensor, const synaptics::synap::Tensor* uv_tensor);

};

SrSublima::SrSublima(): p{new Private}
{
}

SrSublima::~SrSublima()
{
}

bool SrSublima::init(const std::string& lut2d, const std::string& slmodel, const std::string& srmodel,
                  const std::string& slmeta, const std::string& hdrjson, const std::string& srmeta, bool hdr_disable, bool only_y) {

    if (!p->_sublima.init(lut2d, slmodel, slmeta, hdrjson)) {
        LOGE << "init sublima failed";
        return false;
    }
    LOGI << "init Sublima success";

    if (!p->_srnetwork.load_model(srmodel, srmeta)) {
        LOGE << "SuperRes failed to load network: " << slmodel;
        return false;
    }
    LOGI << "init SuperRes success";
    if (hdr_disable) {
        p->_sublima.set_hdr(false);
    }
    if (only_y) {
        p->_sublima.set_only_y(true);
    }
    return true;
}

bool SrSublima::process(const uint8_t* nv12y, const uint8_t* nv12uv, uint8_t* nv15y, uint8_t* nv15uv,
                      uint32_t width_sr, uint32_t height_sr, uint32_t width_sl, uint32_t height_sl, uint32_t nv15_pad) {

    Timer tot;
    uint32_t yIdx = p->_srnetwork.inputs[0].size() > p->_srnetwork.inputs[1].size() ? 0 : 1;
    uint32_t uvIdx = 1 - yIdx;
    LOGI << "SuperRes yIdx=" << yIdx << " SuperRes uvIdx=" << uvIdx; 

    Dimensions dim;
    dim = p->_srnetwork.inputs[yIdx].dimensions();
    size_t y_size = dim.w * dim.h;

    if (!p->_srnetwork.inputs[yIdx].assign(nv12y, y_size)) {
        LOGE << "network set inY buffer failed";
        return false;
    }

    if (!p->_srnetwork.inputs[uvIdx].assign(nv12uv, y_size/2)) {
        LOGE << "network set inUV buffer failed";
        return false;
    }
    p->_superResTimings.assign += tot.get();

    Timer t;
    //SuperRes inference
    bool success = p->_srnetwork.predict();
    p->_superResTimings.infer += t.get();
    p->_superResTimings.tot += tot.get();
    p->_superResTimings.cnt += 1;

    // p->dump_sr_output_nv12("output_check_sr", &p->_srnetwork.outputs[yIdx], &p->_srnetwork.outputs[uvIdx]); //to debug the SR output

    bool success_sublima = p->_sublima.process((uint8_t *)p->_srnetwork.outputs[yIdx].data(), (uint8_t *)p->_srnetwork.outputs[uvIdx].data(),
                                    nv15y, nv15uv,
                                    width_sl, height_sl, nv15_pad);
    LOGI << "sr sublima processing done";
    return true;
}

void SrSublima::Private::dump_sr_output_nv12(const std::string& output_filename_raw, const Tensor* y_tensor, const Tensor* uv_tensor) {

    std::ofstream wf(output_filename_raw, std::ios::out | std::ios::binary);
    if (!wf.is_open()) {
        std::cerr << "Failed to open file for writing: " << output_filename_raw << std::endl;
        return;
    }

    // Get the size of Y and UV planes
    size_t y_size = y_tensor->size();
    size_t uv_size = uv_tensor->size();

    // Write the Y plane data to the file
    wf.write(static_cast<const char*>(y_tensor->data()), y_size);
    wf.write(reinterpret_cast<const char*>(uv_tensor->data()), uv_size);
    wf.close();
    if (!wf.good()) {
        std::cerr << "Error occurred during file write." << std::endl;
    } else {
        std::cout << "Successfully wrote raw NV12 image to " << output_filename_raw << std::endl;
    }
}

SrSublima::Timings* SrSublima::superResTimings()
{
    return &p->_superResTimings;
}

Sublima::Timings* SrSublima::sublimaTimings()
{
    return p->_sublima.timings();
}

Sublima::JsonHDRInfo SrSublima::hdrinfo(){
    return p->_sublima.hdrinfo();
}

}  // namespace synap
}  // namespace synaptics
