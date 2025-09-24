/**************************************************************************//**
 * @file     InferenceProcess.hpp
 * @version  V0.10
 * @brief    Inference process header file
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef INFERENCE_TASK_HPP
#define INFERENCE_TASK_HPP

/* Zephyr, not FreeRTOS */
#if defined(__ZEPHYR__)
#include <zephyr/kernel.h>
#else
#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "task.h"
#endif

#include "DetectorPostProcessing.hpp" /* Post-processing class. */
#include "Model.hpp"

#if defined(__PROFILE__)
    #include "Profiler.hpp"
#endif

using namespace arm::app;

namespace InferenceProcess
{

class InferenceProcess
{
public:
    InferenceProcess(Model *model);
    bool RunJob(
        object_detection::DetectorPostprocessing *pPostProc,
        int modelCols,
        int mode1Rows,
        int srcImgWidth,
        int srcImgHeight,
        std::vector<object_detection::DetectionResult> *results);
protected:

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
#endif

    Model *m_model = nullptr;
};
}// namespace InferenceProcess

struct ProcessTaskParams
{
    Model *model;
    /* On zephyr, use k_queue */
#if defined(__ZEPHYR__)
    k_queue *queueHandle;
#else
    QueueHandle_t queueHandle;
#endif
};

struct xInferenceJob
{
    /* On zephyr, use k_queue */
#if defined(__ZEPHYR__)
    k_queue *responseQueue;
#else
    QueueHandle_t responseQueue;
#endif
    object_detection::DetectorPostprocessing *pPostProc;
    int modelCols;
    int mode1Rows;
    int srcImgWidth;
    int srcImgHeight;

    std::vector<object_detection::DetectionResult> *results;
};

/* On zephyr, use k_queue */
#if defined(__ZEPHYR__)
void inferenceProcessTask(void *pvParameters, void *p2, void *p3);
#else
void inferenceProcessTask(void *pvParameters);
#endif

#endif
