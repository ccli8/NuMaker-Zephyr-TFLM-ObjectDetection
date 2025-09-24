/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Yolo fastest network sample. Demonstrate object detection.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <string>
#include <cinttypes>

#include "BoardInit.hpp"      /* Board initialisation */
/* On zephyr, redirect ml-embedded-evaluation-kit logging to zephyr way */
#if defined(__ZEPHYR__)
#define REGISTER_LOG_MODULE_APP 1
#endif
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "InputFiles.hpp"             /* Baked-in input (not needed for live data) */
#include "Labels.hpp"

#include "InferenceTask.hpp"
#include "DetectorPostProcessing.hpp"
#include "YoloFastestModel.hpp"       /* Model API */

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

/* On zephyr, configure via Kconfig */
#if defined(__ZEPHYR__)
#else
#define __USE_CCAP__
#define __USE_DISPLAY__
//#define __USE_UVC__
#endif

#include "Profiler.hpp"

#if defined (__USE_CCAP__)
    #include "ImageSensor.h"
#endif
#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

/* FreeRTOS only */
#if !defined(__ZEPHYR__)
#define MAINLOOP_TASK_PRIO  4
#define INFERENCE_TASK_PRIO 3
#endif
#define IMAGE_DISP_UPSCALE_FACTOR 1

#if defined(LT7381_LCD_PANEL)
    #define FONT_DISP_UPSCALE_FACTOR 2
#else
    #define FONT_DISP_UPSCALE_FACTOR 1
#endif

#define NUM_FRAMEBUF 2  //1 or 2

typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<object_detection::DetectionResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];


/* FreeRTOS only */
#if !defined(__ZEPHYR__)
#ifdef __cplusplus
extern "C" {
#endif
extern void FreeRTOS_TickHook(uint32_t u32CurrentTickCnt);

#ifdef __cplusplus
}
#endif
#endif

#if defined(__ZEPHYR__)
#include <zephyr/shell/shell.h>
#endif

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace yolofastest
{
	
	/* On zephyr, use shell instead of unsupported getchar */
#if defined(__ZEPHYR__)

K_MUTEX_DEFINE(mutex_record_ctrl);
K_CONDVAR_DEFINE(condvar_record_ctrl);

static bool record_ctrl_cont;
static bool record_ctrl_oneshot;
static bool record_ctrl_end;

static int od_next_cmd_handler(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(sh);
    ARG_UNUSED(argc);
    ARG_UNUSED(argv);

    k_mutex_lock(&mutex_record_ctrl, K_FOREVER);
#if !defined(USE_DMIC)
    record_ctrl_cont = false;
    record_ctrl_oneshot = true;
#else
    record_ctrl_cont = true;
    record_ctrl_oneshot = false;
    shell_print(sh, "Not support one-shot for DMIC\n");
#endif
    k_condvar_signal(&condvar_record_ctrl);
    k_mutex_unlock(&mutex_record_ctrl);

    return 0;
}

static int od_exit_cmd_handler(const struct shell *sh, size_t argc,
			        char **argv)
{
    ARG_UNUSED(sh);
    ARG_UNUSED(argc);
    ARG_UNUSED(argv);

    k_mutex_lock(&mutex_record_ctrl, K_FOREVER);
    record_ctrl_cont = false;
    record_ctrl_oneshot = false;
    record_ctrl_end = true;
    k_condvar_signal(&condvar_record_ctrl);
    k_mutex_unlock(&mutex_record_ctrl);

    return 0;
}

static int od_resume_cmd_handler(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(sh);
    ARG_UNUSED(argc);
    ARG_UNUSED(argv);

    k_mutex_lock(&mutex_record_ctrl, K_FOREVER);
    record_ctrl_cont = true;
    record_ctrl_oneshot = false;
    k_condvar_signal(&condvar_record_ctrl);
    k_mutex_unlock(&mutex_record_ctrl);

    return 0;
}

static int od_suspend_cmd_handler(const struct shell *sh, size_t argc, char **argv)
{
    ARG_UNUSED(sh);
    ARG_UNUSED(argc);
    ARG_UNUSED(argv);

    k_mutex_lock(&mutex_record_ctrl, K_FOREVER);
    record_ctrl_cont = false;
    record_ctrl_oneshot = false;
    k_condvar_signal(&condvar_record_ctrl);
    k_mutex_unlock(&mutex_record_ctrl);

    return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(od_subcmd_set,
	SHELL_CMD_ARG(exit, NULL, "Exit object detection app", od_exit_cmd_handler, 1, 0),
	SHELL_CMD_ARG(next, NULL, "Resume object detection recording one-shot", od_next_cmd_handler, 1, 0),
	SHELL_CMD_ARG(resume, NULL, "Resume object detection recording continuously", od_resume_cmd_handler, 1, 0),
	SHELL_CMD_ARG(suspend, NULL, "Suspend object detection recording", od_suspend_cmd_handler, 1, 0),
	SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(od, &od_subcmd_set, "object detection", NULL);

#endif

extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace yolofastest */
} /* namespace app */
} /* namespace arm */

#if defined(__ZEPHYR__)
#if defined(CONFIG_NVT_ML_HYPERRAM)
#include <zephyr/arch/common/init.h>

extern char __hyperram_data_start[];
extern char __hyperram_data_end[];
extern char __hyperram_bss_start[];
extern char __hyperram_bss_end[];
extern char __hyperram_data_load_start[];

static void HyperRAM_InitCRT(void)
{
    /* Initialize .hyperram.data* sections per load sections */
    memcpy(&__hyperram_data_start,
           &__hyperram_data_load_start,
           __hyperram_data_end - __hyperram_data_start);

    /* Initialize .hyperram.bss* sections to zero */
    memset(&__hyperram_bss_start, 0,
	   (uintptr_t) &__hyperram_bss_end - (uintptr_t) &__hyperram_bss_start);
}
#endif
#endif

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }

    return NULL;
}


/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#else
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#endif

//RGB565
#define IMAGE_FB_SIZE   (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE (1024)

/* On zephyr, allocate at '.nocache.*' sections for non-cache-able */
#if defined(__ZEPHYR__)
__attribute__((section(".nocache.bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".nocache.bss.sram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".nocache.bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif
#else
__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.sram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

static bool PresentInferenceResult(const std::vector<arm::app::object_detection::DetectionResult> &results,
                                   std::vector<std::string> &labels)
{
    /* If profiling is enabled, and the time is valid. */
    info("Final results:\n");

    for (uint32_t i = 0; i < results.size(); ++i)
    {
        info("%" PRIu32 ") %s(%f) -> %s {x=%d,y=%d,w=%d,h=%d}\n", i,
             labels[results[i].m_cls].c_str(),
             results[i].m_normalisedVal, "Detection box:",
             results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h);
    }

    return true;
}


static void DrawImageDetectionBoxes(
    const std::vector<arm::app::object_detection::DetectionResult> &results,
    image_t *drawImg,
    std::vector<std::string> &labels)
{
    for (const auto &result : results)
    {
        imlib_draw_rectangle(drawImg, result.m_x0, result.m_y0, result.m_w, result.m_h, COLOR_B5_MAX, 1, false);
        imlib_draw_string(drawImg, result.m_x0, result.m_y0 - 16, labels[result.m_cls].c_str(), COLOR_B5_MAX, 2, 0, 0, false,
                          false, false, false, 0, false, false);
    }
}

static void main_task(void *pvParameters)
{
    BaseType_t ret;

    info("main task running \n");
    /* Model object creation and initialisation. */
    arm::app::YoloFastestModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::yolofastest::GetModelPointer(),
                    arm::app::yolofastest::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        vTaskDelete(nullptr);
        return;
    }

    /*
     * On zephyr, configure mpu region for tensor arena in zephyr way
     * rather than direct control
     *
     * With CONFIG_CONFIG_NOCACHE_MEMORY=y, frame buffer is allocated
     * at non-cache-able memory.
     */
#if !defined(__ZEPHYR__)
    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - cacheable WTRA
        },
#if defined (__USE_CCAP__)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#if (NUM_FRAMEBUF == 2)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#endif
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());
#endif

    // Setup inference resource and create task
    struct ProcessTaskParams taskParam;
    QueueHandle_t inferenceProcessQueue = xQueueCreate(1, sizeof(xInferenceJob *));
    QueueHandle_t inferenceResponseQueue = xQueueCreate(1, sizeof(xInferenceJob *));

    taskParam.model = &model;
    taskParam.queueHandle = inferenceProcessQueue;

    ret = xTaskCreate(inferenceProcessTask, "inference task", 2 * 1024, &taskParam, INFERENCE_TASK_PRIO, nullptr);

    if (ret != pdPASS)
    {
        printf_err("FreeRTOS: Failed to inference process task \n");
        vTaskDelete(nullptr);
        return;
    }

#if !defined (__USE_CCAP__)
    uint8_t u8ImgIdx = 0;
    char chStdIn;
#endif

    TfLiteTensor *inputTensor   = model.GetInputTensor(0);
    TfLiteTensor *outputTensor = model.GetOutputTensor(0);

    if (!inputTensor->dims)
    {
        printf_err("Invalid input tensor dims\n");
        vTaskDelete(nullptr);
        return;
    }
    else if (inputTensor->dims->size < 3)
    {
        printf_err("Input tensor dimension should be >= 3\n");
        vTaskDelete(nullptr);
        return;
    }

    TfLiteIntArray *inputShape = model.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::YoloFastestModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::YoloFastestModel::ms_inputRowsIdx];

    // postProcess
    arm::app::object_detection::DetectorPostprocessing postProcess(0.5, 0.45, numClasses, 0);

    //label information
    std::vector<std::string> labels;
    GetLabelsVector(labels);

    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle = 0;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
    info("init perfcycles %llu \n", u64PerfCycle);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    struct xInferenceJob *inferenceJob = new (struct xInferenceJob);

#if defined (__USE_CCAP__)
    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);
#endif

#if defined (__USE_DISPLAY__)
    char szDisplayText[160];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
    UVC_Init();
    HSUSBD_Start();
#endif

    while (1)
    {

        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
            /* Detector post-processing*/

            inferenceJob->responseQueue = inferenceResponseQueue;
            inferenceJob->pPostProc = &postProcess;
            inferenceJob->modelCols = inputImgCols;
            inferenceJob->mode1Rows = inputImgRows;
            inferenceJob->srcImgWidth = infFramebuf->frameImage.w;
            inferenceJob->srcImgHeight = infFramebuf->frameImage.h;
            inferenceJob->results = &infFramebuf->results; //&results;

            xQueueReceive(inferenceResponseQueue, &inferenceJob, portMAX_DELAY);
        }

        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            //resize full image to input tensor
            image_t resizeImg;

            roi.x = 0;
            roi.y = 0;
            roi.w = fullFramebuf->frameImage.w;
            roi.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols;
            resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
            resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            /* If the data is signed. */
            if (model.IsDataSigned())
            {
                arm::app::image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
            }

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
            //trigger inference
            inferenceJob->responseQueue = inferenceResponseQueue;
            inferenceJob->pPostProc = &postProcess;
            inferenceJob->modelCols = inputImgCols;
            inferenceJob->mode1Rows = inputImgRows;
            inferenceJob->srcImgWidth = fullFramebuf->frameImage.w;
            inferenceJob->srcImgHeight = fullFramebuf->frameImage.h;
            inferenceJob->results = &fullFramebuf->results;

            xQueueSend(inferenceProcessQueue, &inferenceJob, portMAX_DELAY);
            fullFramebuf->eState = eFRAMEBUF_INF;
        }

        if (infFramebuf)
        {
            //draw bbox and render
            /* Draw boxes. */
            DrawImageDetectionBoxes(infFramebuf->results, &infFramebuf->frameImage, labels);

            //display result image
#if defined (__USE_DISPLAY__)
            //Display image on LCD
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = ((infFramebuf->frameImage.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
            sDispRect.u32BottonRightY = ((infFramebuf->frameImage.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)

            if (UVC_IsConnect())
            {
#if (UVC_Color_Format == UVC_Format_YUY2)
                rectangle_t roi;

                image_t RGB565Img;
                image_t YUV422Img;

                RGB565Img.w = infFramebuf->frameImage.w;
                RGB565Img.h = infFramebuf->frameImage.h;
                RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
                RGB565Img.pixfmt = PIXFORMAT_RGB565;

                YUV422Img.w = RGB565Img.w;
                YUV422Img.h = RGB565Img.h;
                YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
                YUV422Img.pixfmt = PIXFORMAT_YUV422;

                roi.x = 0;
                roi.y = 0;
                roi.w = RGB565Img.w;
                roi.h = RGB565Img.h;
                imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);

#else
                image_t origImg;
                image_t vflipImg;

                origImg.w = infFramebuf->frameImage.w;
                origImg.h = infFramebuf->frameImage.h;
                origImg.data = (uint8_t *)infFramebuf->frameImage.data;
                origImg.pixfmt = PIXFORMAT_RGB565;

                vflipImg.w = origImg.w;
                vflipImg.h = origImg.h;
                vflipImg.data = (uint8_t *)infFramebuf->frameImage.data;
                vflipImg.pixfmt = PIXFORMAT_RGB565;

                imlib_nvt_vflip(&origImg, &vflipImg);
#endif
                UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);

            }

#endif

            u64PerfFrames ++;

            if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //              sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);

                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = (frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR);
                sDispRect.u32BottonRightX = (frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR);
                sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);
                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
                    C_BLUE,
                    C_WHITE,
                    false,
                    FONT_DISP_UPSCALE_FACTOR
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
            }

            PresentInferenceResult(infFramebuf->results, labels);
            infFramebuf->eState = eFRAMEBUF_EMPTY;
        }

        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
#if !defined (__USE_CCAP__)
            info("Press 'n' to run next image inference \n");
            info("Press 'q' to exit program \n");

            while ((chStdIn = getchar()))
            {
                if (chStdIn == 'q')
                {
                    vTaskDelete(nullptr);
                    return;
                }
                else if (chStdIn != 'n')
                {
                    break;
                }
            }

            const uint8_t *pu8ImgSrc = get_img_array(u8ImgIdx);

            if (nullptr == pu8ImgSrc)
            {
                printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", u8ImgIdx,
                           NUMBER_OF_FILES - 1);
                vTaskDelete(nullptr);
                return;
            }

            u8ImgIdx ++;

            if (u8ImgIdx >= NUMBER_OF_FILES)
                u8ImgIdx = 0;

#endif

#if defined (__USE_CCAP__)
            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

            ImageSensor_Capture((uint32_t)(emptyFramebuf->frameImage.data));

#if defined(__PROFILE__)
            u64CCAPEndCycle = pmu_get_systick_Count();
            info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif

#else
            //copy source image to frame buffer
            image_t srcImg;

            srcImg.w = IMAGE_WIDTH;
            srcImg.h = IMAGE_HEIGHT;
            srcImg.data = (uint8_t *)pu8ImgSrc;
            srcImg.pixfmt = PIXFORMAT_RGB888;

            roi.x = 0;
            roi.y = 0;
            roi.w = IMAGE_WIDTH;
            roi.h = IMAGE_HEIGHT;

            imlib_nvt_scale(&srcImg, &emptyFramebuf->frameImage, &roi);
#endif
            emptyFramebuf->results.clear();
            emptyFramebuf->eState = eFRAMEBUF_FULL;
        }

        vTaskDelay(1);
    }

}

int main()
{
    BaseType_t ret;

    /* Initialize the UART module to allow printf related functions (if using retarget) */
    BoardInit();

#if defined(__ZEPHYR__)
#if defined(CONFIG_NVT_ML_HYPERRAM)
    HyperRAM_InitCRT();
#endif
#endif

    /* On zephyr, scheduler has started before run to main */
#if defined(__ZEPHYR__)
    main_task(nullptr);
#else
    /* Create main task. */
    ret = xTaskCreate(main_task, "main task", 2 * 1024, nullptr, MAINLOOP_TASK_PRIO, nullptr);

    if (ret != pdPASS)
    {
        info("FreeRTOS: Failed to create main look task \n");
        return -1;
    }

    // Start Scheduler
    vTaskStartScheduler();

    /* This is unreachable without errors. */
    info("program terminating...\n");
#endif
}

/*-----------------------------------------------------------*/

/* FreeRTOS only */
#if !defined(__ZEPHYR__)

void vApplicationMallocFailedHook(void)
{
    /* vApplicationMallocFailedHook() will only be called if
    configUSE_MALLOC_FAILED_HOOK is set to 1 in FreeRTOSConfig.h.  It is a hook
    function that will get called if a call to pvPortMalloc() fails.
    pvPortMalloc() is called internally by the kernel whenever a task, queue,
    timer or semaphore is created.  It is also called by various parts of the
    demo application.  If heap_1.c or heap_2.c are used, then the size of the
    heap available to pvPortMalloc() is defined by configTOTAL_HEAP_SIZE in
    FreeRTOSConfig.h, and the xPortGetFreeHeapSize() API function can be used
    to query the size of free heap space that remains (although it does not
    provide information on how the remaining heap might be fragmented). */
    info("vApplicationMallocFailedHook\n");
    taskDISABLE_INTERRUPTS();

    for (;;);
}
/*-----------------------------------------------------------*/

void vApplicationIdleHook(void)
{
    /* vApplicationIdleHook() will only be called if configUSE_IDLE_HOOK is set
    to 1 in FreeRTOSConfig.h.  It will be called on each iteration of the idle
    task.  It is essential that code added to this hook function never attempts
    to block in any way (for example, call xQueueReceive() with a block time
    specified, or call vTaskDelay()).  If the application makes use of the
    vTaskDelete() API function (as this demo application does) then it is also
    important that vApplicationIdleHook() is permitted to return to its calling
    function, because it is the responsibility of the idle task to clean up
    memory allocated by the kernel to any task that has since been deleted. */
}
/*-----------------------------------------------------------*/

void vApplicationStackOverflowHook(TaskHandle_t pxTask, char *pcTaskName)
{
    (void) pcTaskName;
    (void) pxTask;

    /* Run time stack overflow checking is performed if
    configCHECK_FOR_STACK_OVERFLOW is defined to 1 or 2.  This hook
    function is called if a stack overflow is detected. */
    info("vApplicationStackOverflowHook\n");
    taskDISABLE_INTERRUPTS();

    //__BKPT();

    //printf("Stack overflow task name=%s\n", pcTaskName);


    for (;;);



}
/*-----------------------------------------------------------*/

void vApplicationTickHook(void)
{
    /* This function will be called by each tick interrupt if
    configUSE_TICK_HOOK is set to 1 in FreeRTOSConfig.h.  User code can be
    added here, but the tick hook is called from an interrupt context, so
    code must not attempt to block, and only the interrupt safe FreeRTOS API
    functions can be used (those that end in FromISR()).  The code in this
    tick hook implementation is for demonstration only - it has no real
    purpose.  It just gives a semaphore every 50ms.  The semaphore unblocks a
    task that then toggles an LED.  Additionally, the call to
    vQueueSetAccessQueueSetFromISR() is part of the "standard demo tasks"
    functionality. */

    FreeRTOS_TickHook(xTaskGetTickCount());
}

#endif

/*-----------------------------------------------------------*/


