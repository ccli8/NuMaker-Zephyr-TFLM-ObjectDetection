Porting notes on this example
#############################


Notes on porting from BSP to Zephyr
***********************************

1. Search keywords below for relevant modifications

   - __ZEPHYR__
   - CONFIG_SOC_FAMILY_NUMAKER

2. Must enable FPU (CONFIG_FPU and friends) additionally to match BSP
   performance on inference pre-process/post-process
3. Configure MPU region for tensor arena in zephyr way rather than
   direct control
4. Use default ethosu_flush_dcache/ethosu_invalidate_dcache override provided
   in ethos-u zephyr driver
5. Profile/PMU

   (1) ethosu_driver is encapsulated in ethos-u zephyr driver and is
       invisible from app. Acquire it by invoking ethosu_reserve_driver
       and ethosu_release_driver in pair transiently
   (2) SysTick is exclusively used by zephyr kernel. Use zephyr kernel
       timing api instead

6. ml-embedded-evaluation-kit

   (1) Change BufAttributes.hpp for model related code/data placement

       a. Place model at .rodata.tflm_model section
       b. Place tensor arena at .noinit.tflm_arena/
          .hyperram.noinit.tflm_arena section
       c. Place input feature map at .rodata.tflm_input section
       d. Place labels at .rodata.tflm_labels section

   (2) Redirect ml-embedded-evaluation-kit logging to zephyr way
       NOTE: Zephyr logging doesn't open interface for disabling newline
       print, and this redirect has drawback of duplicate newline print.

7. ml-embedded-evaluation-kit/object_detection

   (1) Remove YoloFastestModel.hpp/YoloFastestModel.cpp. Use the versions
       provided in M55M1 BSP ObjectDetection_FreeRTOS/Model

8. openmv

   (1) Base on M55M1 BSP V3.01.002 ThirdParty/openmv
   (2) Remove GCC/IAR/KEIL IDE projects and their build artifacts (*.a/*.lib)
   (3) Port mutex onto zephyr (N/A)
   (4) Remove unnecessary sensors/ from build (N/A)

9. Use shell instead of unsupported getchar
   Add shell commands "od <subcommand>" for object detection record control

10. Adjust memory size

    (1) Enlarge main stack size (CONFIG_MAIN_STACK_SIZE)
        NOTE: MPU fault may imply stack overflow.
    (2) Enable system heap (CONFIG_HEAP_MEM_POOL_SIZE)
        This is used for k_malloc in ethos-u driver overrides
        ethosu_mutex_create and ethosu_semaphore_create

11. Add more tflite model blobs
    (1) Non-vela-compile
        NOTE: Its source comes from below:
        https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolo_fastest_v1.1
    (2) Vela-compile targeting ethos-u55/256 MAC, optimize for performance
        NOTE: It is copied from M55M1 BSP ObjectDetection_FreeRTOS/Model
    (3) Vela-compile targeting ethos-u55/256 MAC, optimize for size

12. HyperRAM
    By this, tensor arena can configure to locate at HyperRAM for model
    needing high memory footprint.

    NOTE: Driver source is M55M1 BSP V3.01.002 SampleCode/StdDriver/
    SPIM_HYPER_ExeInHRAM, not ObjectDetection_FreeRTOS/Device/HyperRAM.
    On zephyr, the ObjectDetection_FreeRTOS HyperRAM driver doesn't
    work somehow.
