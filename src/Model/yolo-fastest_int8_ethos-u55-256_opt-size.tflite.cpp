/*
 * Copyright (c) 2024, Arm Limited and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Vela-compile tflite model
 *
 * Target ethos-u55/256 MAC (--accelerator-config ethos-u55-256)
 * Optimize for size (--optimise Size)
 */

#include "BufAttributes.hpp"

#include <cstddef>
#include <cstdint>

extern const int originalImageSize = 320;
extern const int channelsImageDisplayed = 3;
extern const float anchor1[] = {12, 18, 37, 49, 52, 132};
extern const float anchor2[] = {115, 73, 119, 199, 242, 238};
extern const int numClasses = 80;

namespace arm
{
namespace app
{
namespace yolofastest
{

static const uint8_t nn_model[] MODEL_TFLITE_ATTRIBUTE =
{
#include "yolo-fastest_int8_ethos-u55-256_opt-size.tflite.inc"
};


const uint8_t *GetModelPointer()
{
    return nn_model;
}

size_t GetModelLen()
{
    return sizeof(nn_model);
}

} /* namespace arm */
} /* namespace app */
} /* namespace yolofastest */
