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

   (3) For ``%p`` in ``printf``, cast pointer type to ``void *`` to be safe
       and suppress compiler warning

7. ml-embedded-evaluation-kit/object_detection

   (1) Remove YoloFastestModel.hpp/YoloFastestModel.cpp. Use the versions
       provided in M55M1 BSP ObjectDetection_FreeRTOS/Model

8. OpenMV

   (1) Base on M55M1 BSP V3.01.002 ThirdParty/openmv
   (2) Remove GCC/IAR/KEIL IDE projects and their build artifacts (*.a/*.lib)
   (3) Port mutex onto zephyr (N/A)
   (4) Remove unnecessary sensors/ from build (N/A)
   (5) Patch imlib_nvt.c by pre-V3.01.005 to adjust '__GNUC__' conditional for
       different GCC versions

9. Use shell instead of unsupported getchar
   Add shell commands "od <subcommand>" for object detection record control

10. Adjust memory size

    (1) Enlarge main stack size (CONFIG_MAIN_STACK_SIZE)
        NOTE: MPU fault may imply stack overflow.
    (2) Enable system heap (CONFIG_HEAP_MEM_POOL_SIZE)
        This is used for k_malloc in ethos-u driver overrides
        ethosu_mutex_create and ethosu_semaphore_create

11. Model

    (1) Add more models

        a. Non-vela-compile
           NOTE: Its source comes from below:
           https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolo_fastest_v1.1
        b. Vela-compile targeting ethos-u55/256 MAC, optimize for performance
           NOTE: It is re-vela-compile of M55M1 BSP ObjectDetection_FreeRTOS/Model
        c. Vela-compile targeting ethos-u55/256 MAC, optimize for size

    (2) Version model blob in binary format
        Model blob is converted to C array data format for inclusion at build
	time. This can reduce versioned file size to 1/6.

12. HyperRAM

    (1) Must change board to NuMaker-X-M55M1, or unstable
    (2) Update HyperRAM driver to V3.02.003 from V3.02.002. It fixes:

        a. Boot and ICE connect may always fail

	   This results from failed MLDO adjustment.
	   It must be done in power stable condition.
	   This can recover by adding boot delay, like the flow below
	   to successfully boot or for ICE connect:

	   Press RESET button > Re-power > Release RESET button

        b. Related to above, the second or third ``west flash`` or ``pyocd reset`` will fail

	   This results from failed MLDO adjustment.
	   ``SYS_MLDOTCTL`` won't reset with each reset, except power-on or chip reset.

    (3) Tensor arena can configure to locate at HyperRAM for model
        needing high memory usage.

13. Display

    (1) Support list

        a. LCD LT7381 (EBI)

    (2) Not support list

        a. LCD FSA506 (EBI)
        b. LCD ILI9341 (SPI)

    (3) Modify board_config.h to enable Kconfig with display choice

    (4) Unlike EBI RAM, EBI LCD needs one MPU region configured to Device
        type (not Normal type) for mapped bank, because its driver is like
        MMIO (not RAM).

14. CCAP
    For wait for capture done, use WAIT_FOR instead and also avoid busy-wait.

15. Performance
    Compiler optimization level defaults to CONFIG_SIZE_OPTIMIZATIONS (-Os).
    To match BSP ObjectDetection_FreeRTOS, configure CONFIG_SPEED_OPTIMIZATIONS
    (-O2).
