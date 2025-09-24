# Copyright (c) 2025 Nuvoton Technology Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(NuMaker-Zephyr-TFLM-ObjectDetection)

target_compile_definitions(app
  PRIVATE
    # Required by app for dcache conditional code
    $<$<BOOL:${CONFIG_CACHE_MANAGEMENT}>:NVT_DCACHE_ON>
    # Required by app to enable CCAP
    $<$<BOOL:${CONFIG_NVT_ML_OD_INPUT_CCAP}>:__USE_CCAP__>
    # Required by app to enable display
    $<$<BOOL:${CONFIG_NVT_ML_OD_OUTPUT_DISPLAY}>:__USE_DISPLAY__>
    # Required by app to enable Ethos-U profiling
    $<$<BOOL:${CONFIG_NVT_ML_ETHOS_U_PROFILE}>:__PROFILE__>
    # Required by ml-embedded-evaluation-kit to go Ethos-U way
    $<$<BOOL:${CONFIG_NVT_ML_REQUIRES_ETHOS_U}>:ARM_NPU>
    # Required by TFLM to define tensor arena size
    ACTIVATION_BUF_SZ=${CONFIG_NVT_ML_TFLM_TENSOR_ARENA_SIZE}
)

set(APP_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(app
  PRIVATE
    ${APP_SOURCE_DIR}/Device/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/application_api_common/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/application_api_use_case_object_detection/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/application_main/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/log/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/math/include
    ${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/profiler/include
    ${APP_SOURCE_DIR}/Model/include
    ${APP_SOURCE_DIR}/NPU/include
    ${APP_SOURCE_DIR}/openmv_clone/omv/alloc
    ${APP_SOURCE_DIR}/openmv_clone/omv/common
    ${APP_SOURCE_DIR}/openmv_clone/omv/imlib
    ${APP_SOURCE_DIR}/openmv_clone/omv/Lib
    ${APP_SOURCE_DIR}/Pattern/include
    ${APP_SOURCE_DIR}/ProfilerCounter/include
    ${APP_SOURCE_DIR}
)

# Add all BSP_Clone *.cc/*.cpp/*.c files
if(CONFIG_SOC_SERIES_M55M1X)
  file(GLOB_RECURSE SOURCE_BSP_CLONE
    "${APP_SOURCE_DIR}/BSP_Clone/M55M1_StdDriver/*.cc"
    "${APP_SOURCE_DIR}/BSP_Clone/M55M1_StdDriver/*.cpp"
    "${APP_SOURCE_DIR}/BSP_Clone/M55M1_StdDriver/*.c"
  )
endif()

# Add all Device *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_DEVICE
    "${APP_SOURCE_DIR}/Device/*.cc"
    "${APP_SOURCE_DIR}/Device/*.cpp"
    "${APP_SOURCE_DIR}/Device/*.c"
)
# Exclude */ImageSensor/*.c* files if not enabled
if(NOT CONFIG_NVT_ML_OD_INPUT_CCAP)
    list(FILTER SOURCE_DEVICE EXCLUDE REGEX ".*/ImageSensor/.*\\.c.*$")
endif()
# Exclude */Display/*.c* files if not enabled
if(NOT CONFIG_NVT_ML_OD_OUTPUT_DISPLAY)
    list(FILTER SOURCE_DEVICE EXCLUDE REGEX ".*/Display/.*\\.c.*$")
endif()
# Exclude */HyperRAM/*.c* files if not enabled
if(NOT CONFIG_NVT_ML_HYPERRAM)
    list(FILTER SOURCE_DEVICE EXCLUDE REGEX ".*/HyperRAM/.*\\.c.*$")
endif()

# Add all ml-embedded-evaluation-kit_clone *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_ML_EMBEDDED_EVALUATION_KIT_CLONE
    "${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/*.cc"
    "${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/*.cpp"
    "${APP_SOURCE_DIR}/ml-embedded-evaluation-kit_clone/*.c"
)

# Add all Model *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_MODEL
    "${APP_SOURCE_DIR}/Model/*.cc"
    "${APP_SOURCE_DIR}/Model/*.cpp"
    "${APP_SOURCE_DIR}/Model/*.c"
)
# Exclude all */Model/*.tflite.c* files from build, except target-specific
list(FILTER SOURCE_MODEL EXCLUDE REGEX ".*/Model/.*\\.tflite\\.c.*$")
if(CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8)
    list(APPEND SOURCE_MODEL "${APP_SOURCE_DIR}/Model/yolo-fastest_int8.tflite.cpp")
elseif(CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8_ETHOS_U55_256_SIZE)
    list(APPEND SOURCE_MODEL "${APP_SOURCE_DIR}/Model/yolo-fastest_int8_ethos-u55-256_opt-size.tflite.cpp")
elseif(CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8_ETHOS_U55_256_SPEED)
    list(APPEND SOURCE_MODEL "${APP_SOURCE_DIR}/Model/yolo-fastest_int8_ethos-u55-256_opt-speed.tflite.cpp")
endif()

# Add all NPU *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_NPU
    "${APP_SOURCE_DIR}/NPU/*.cc"
    "${APP_SOURCE_DIR}/NPU/*.cpp"
    "${APP_SOURCE_DIR}/NPU/*.c"
)
# Exclude ethosu*.c* or ethos_u*.c* files if not enabled
if(NOT CONFIG_NVT_ML_REQUIRES_ETHOS_U)
    list(FILTER SOURCE_NPU EXCLUDE REGEX ".*ethos_?u.*\\.c.*$")
endif()

# Make openmv available for image processing, even though camera
# and display are not enabled.
# Add all openmv *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_OPENMV
    "${APP_SOURCE_DIR}/openmv_clone/*.cc"
    "${APP_SOURCE_DIR}/openmv_clone/*.cpp"
    "${APP_SOURCE_DIR}/openmv_clone/*.c"
)
# Exclude all */omv/common/*.c* files from build, except array.c
# and mutex.c
list(FILTER SOURCE_OPENMV EXCLUDE REGEX ".*/omv/common/.*\\.c.*$")
list(APPEND SOURCE_OPENMV
    "${APP_SOURCE_DIR}/openmv_clone/omv/common/array.c"
    "${APP_SOURCE_DIR}/openmv_clone/omv/common/mutex.c"
)
# Exclude unnecessary */omv/sensors/*.c* files from build
list(FILTER SOURCE_OPENMV EXCLUDE REGEX ".*/omv/sensors/.*\\.c.*$")

# Add all Pattern *.cc/*.cpp/*.c files
file(GLOB_RECURSE SOURCE_PATTERN
    "${APP_SOURCE_DIR}/Pattern/*.cc"
    "${APP_SOURCE_DIR}/Pattern/*.cpp"
    "${APP_SOURCE_DIR}/Pattern/*.c"
)

if(CONFIG_NVT_ML_ETHOS_U_PROFILE)
    # Add all ProfilerCounter *.cc/*.cpp/*.c files
    file(GLOB_RECURSE SOURCE_PROFILER_COUNTER
        "${APP_SOURCE_DIR}/ProfilerCounter/*.cc"
        "${APP_SOURCE_DIR}/ProfilerCounter/*.cpp"
        "${APP_SOURCE_DIR}/ProfilerCounter/*.c"
    )
else()
    # Make pmu_get_systick_Count available for display control
    list(APPEND SOURCE_PROFILER_COUNTER
        "${APP_SOURCE_DIR}/ProfilerCounter/pmu_counter.c"
    )
endif()

# Add root *.cc/*.cpp/*.c files
file(GLOB SOURCE_ROOT
    "${APP_SOURCE_DIR}/*.cc"
    "${APP_SOURCE_DIR}/*.cpp"
    "${APP_SOURCE_DIR}/*.c"
)

target_sources(app
  PRIVATE
    ${SOURCE_BSP_CLONE}
    ${SOURCE_DEVICE}
    ${SOURCE_ML_EMBEDDED_EVALUATION_KIT_CLONE}
    ${SOURCE_MODEL}
    ${SOURCE_NPU}
    ${SOURCE_OPENMV}
    ${SOURCE_PATTERN}
    ${SOURCE_PROFILER_COUNTER}
    ${SOURCE_ROOT}
)

# Add HyperRAM sections
zephyr_linker_sources(SECTIONS hyperram.ld)
