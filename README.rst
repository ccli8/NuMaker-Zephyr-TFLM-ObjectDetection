Example for Object Detection on Nuvoton's Ethos-U NPU platform
##############################################################

Overview
********

This example shows one object detection inference application,
which detects and draws face bounding box in a given image,
on Nuvoton's Ethos-U NPU capable platform.
Its very source comes from
Arm's `ML embedded evaluation kit/object detection`_ example,
which has more information on the object detection application.

.. _ML embedded evaluation kit/object detection: https://gitlab.arm.com/artificial-intelligence/ethos-u/ml-embedded-evaluation-kit/-/blob/main/docs/use_cases/object_detection.md

This example mainly consists of:

- Optionally, capture video frame through CCAP
- Run object detection inference using the `TensorFlow Lite Micro`_ framework
  and the Ethos-U NPU
- Optionally, output the inference result on display device

This example runs a model `YOLO-Fastest v1.1`_ [#]_
that comes from the `ML_ZOO`_.
This model has been converted to ``tflite`` format, named `yolo-fastest_int8.tflite`_.
It then can be optimized using the `Vela compiler`_ [#]_.

.. [#] Check out `Choosing object detection model`_
.. [#] Check out `Optimizing model using Vela`_

.. _TensorFlow Lite Micro: https://github.com/tensorflow/tflite-micro
.. _ML_ZOO: https://github.com/OpenNuvoton/ML_YOLO
.. _YOLO-Fastest v1.1: https://github.com/OpenNuvoton/ML_YOLO/tree/master/yolo_fastest_v1.1
.. _yolo-fastest_int8.tflite: src/Model/yolo-fastest_int8.tflite

Vela takes a ``tflite`` file as input and produces another ``tflite`` file as output,
where the operators supported by Ethos-U have been replaced by an Ethos-U custom operator.
In an ideal case the complete network would be replaced by a single Ethos-U custom operator.

Support targets
===============

+----------------------+------------------+---------------------+
| Board                | Zephyr target    |NPU                  |
+======================+==================+=====================+
| `NuMaker-X-M55M1`_   | `numaker_m55m1`_ |Ethos-U55 256 MAC    |
+----------------------+------------------+---------------------+

.. _NuMaker-X-M55M1: https://direct.nuvoton.com/tw/numaker-x-m55m1d
.. _numaker_m55m1: https://docs.zephyrproject.org/latest/boards/nuvoton/numaker_m55m1/doc/index.html
.. _NuMaker-X-M55M1 board: `NuMaker-X-M55M1`_

Hardware requirements
=====================

- `NuMaker-X-M55M1 board`_

  This example needs to build and run on Ethos-U NPU capable platform.
  In this document, `NuMaker-X-M55M1 board`_ is taken for demo.

- CMOS image sensor HM1055 (optional)

- LCD LT7381 (optional)

Software requirements
=====================

- Host operating system: Windows 10 64-bit or afterwards

  Most users of Nuvoton's Cortex-M series SoC develop on Windows,
  so this document favors this environment.

  The command lines in this document are verified on Windows Git Bash environment.
  For other shell environments, check on how differently shells use line continuation,
  quotation marks, and escapes characters.
  For Bash, line continuation mark is "\\".

- `Zephyr development environment`_

  .. _Zephyr development environment: https://docs.zephyrproject.org/latest/develop/index.html

- `Git`_

  This document favors Git Bash as CLI environment.

  .. _Git: https://git-scm.com/

- Cross GCC compiler

  Use `Zephyr SDK toolchain`_ instead of `Arm GNU Toolchain`_ to avoid build failure
  caused by toolchain difference.

  .. _Zephyr SDK toolchain: https://docs.zephyrproject.org/latest/develop/getting_started/index.html#install-the-zephyr-sdk
  .. _Arm GNU Toolchain: https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain
  
- `OpenNuvoton pyOCD`_

  The PyPI pyOCD support for M55M1 hasn't been ready.
  Install from `OpenNuvoton pyOCD`_ instead:

  .. code-block:: console

     $ pip uninstall pyocd
     $ pip install git+https://github.com/OpenNuvoton/pyOCD

  Confirm pyOCD version is ``0.36`` or afterwards:

  .. code-block:: console

     $ pyocd --version
     0.36.1.dev3

  .. _OpenNuvoton pyOCD: https://github.com/OpenNuvoton/pyOCD

- `Vela compiler`_ (`PyPI package`__) (optional)

  Use for `Optimizing model using Vela`_.

  .. _Vela compiler: https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela
  .. __: https://pypi.org/project/ethos-u-vela/

Building and Running
********************

Building the example
====================

This example doesn't upstream to Zephyr mainline, and exits as `Zephyr workspace application`_.
Assuming that the `zephyrproject` West workspace and the `zephyrproject/zephyr`
repository have settled via `above`__, clone this example:

.. _Zephyr workspace application: https://docs.zephyrproject.org/latest/develop/application/index.html#zephyr-workspace-application

__ `Zephyr development environment`_

.. code-block:: console

   $ cd zephyrproject
   $ mkdir applications
   $ cd applications
   $ git clone https://github.com/OpenNuvoton/NuMaker-Zephyr-TFLM-ObjectDetection
   $ cd ..
    
Now, we get back to `zephyrproject`.
Add the `tflite-micro` module to your West manifest and pull it:

.. code-block:: console

   $ west config manifest.project-filter -- +tflite-micro
   $ west update

Dependent on configuration options, we can:

- Build the example with CCAP and display disabled by default:

  .. code-block:: console

     $ west -v build \
     -b numaker_m55m1 \
     applications/NuMaker-Zephyr-TFLM-ObjectDetection

- Build the example with sensor HM1055 and LCD LT7381 enabled:

  .. code-block:: console

     $ west -v build \
     -b numaker_m55m1 \
     applications/NuMaker-Zephyr-TFLM-ObjectDetection \
     --shield himax_hm1055 \
     --shield levetop_lt7381

Flash the generated image:

.. code-block:: console

    $ west flash

Monitoring the example
======================

To monitor the example, we need to:

- Configure host terminal program with **115200/8-N-1**

After running the example via ``west flash``, on host terminal,
you should see messages like:

.. code-block:: console
   
   <inf> app: BoardInit: complete
   <inf> app: Target system: M55M1
   <inf> app: main task running
   <inf> app: Added ethos-u support to op resolver
   <inf> app: Creating allocator using tensor arena at 0x0x2014d520

The inference model's information is shown:

.. code-block:: console

   <inf> app: Allocating tensors
   <inf> app: Model INPUT tensors:
   <inf> app:   tensor type is INT8
   <inf> app:   tensor occupies 307200 bytes with dimensions
   <int> app:           0:   1
   <inf> app:           1: 320
   <inf> app:           2: 320
   <inf> app:           3:   3
   <inf> app: Quant dimension: 0
   <inf> app: Scale[0] = 0.003922
   <inf> app: ZeroPoint[0] = -128
   <inf> app: Model OUTPUT tensors:
   <inf> app:   tensor type is INT8
   <inf> app:   tensor occupies 102000 bytes with dimensions
   <inf> app:           0:   1
   <inf> app:           1:  20
   <inf> app:           2:  20
   <inf> app:           3: 255
   <inf> app: Quant dimension: 0
   <inf> app: Scale[0] = 0.190095
   <inf> app: ZeroPoint[0] = 77
   <inf> app:   tensor type is INT8
   <inf> app:   tensor occupies 25500 bytes with dimensions
   <inf> app:           0:   1
   <inf> app:           1:  10
   <inf> app:           2:  10
   <inf> app:           3: 255
   <inf> app: Quant dimension: 0
   <inf> app: Scale[0] = 0.220456
   <inf> app: ZeroPoint[0] = 67
   <inf> app: Activation buffer (a.k.a tensor arena) size used: 840328
   <inf> app: Number of operators: 1
   <inf> app:   Operator 0: ethos-u

- With CCAP and display disabled, we can run the following commands
  in host terminal program to control object detection flow:

  .. code-block:: console

     <wrn> app: Press 'od next' to resume object detection inference one-shot
     <wrn> app: Press 'od resume' to resume object detection inference continuously
     <wrn> app: Press 'od suspend' to suspend object detection inference
     <wrn> app: Press 'od exit' to exit program

- With CCAP and display enabled, we can see inference output directly
  on display device.

Further reading
***************

Optimizing model using Vela
===========================

This section instructs how to optimize a model using Vela.
We take `yolo-fastest_int8.tflite`_ as example model to optimize using Vela
and M55M1 Ethos-U NPU as target for which to optimize:

.. important:: M55M1 Ethos-U NPU is Ethos-U55, 256 macs_per_cycle.
   The configuration value `ethos-u55-256` must match.

1. Locate `yolo-fastest_int8.tflite`_ in this example directory,
   create a separate directory named e.g. ``yolo-fastest_int8``,
   and place this file in this directory.

2. Optimize the model `yolo-fastest_int8.tflite`_ using Vela compiler

   We can optimize for size or speed.
   After Vela-compile, we get optimized model named ``yolo-fastest_int8_vela.tflite``. 
   Then rename this file to match its semantics of optimization option.

   .. _Optimizing for size:

   - Optimizing for size

     .. code-block:: console

        $ cd yolo-fastest_int8
        $ vela fastest_int8.tflite \
        --output-dir . \
        --accelerator-config ethos-u55-256 \
        --optimise Size
        $ mv yolo-fastest_int8_vela.tflite yolo-fastest_int8_ethos-u55-256_opt-size.tflite

     .. note:: The renamed model file is just how
        `yolo-fastest_int8_ethos-u55-256_opt-size.tflite`_
        in this example directory is generated.

     .. _yolo-fastest_int8_ethos-u55-256_opt-size.tflite: src/Model/yolo-fastest_int8_ethos-u55-256_opt-size.tflite

   .. _Optimizing for speed:

   - Optimizing for speed
   
     .. code-block:: console

        $ cd yolo-fastest_int8
        $ vela fastest_int8.tflite \
        --output-dir . \
        --accelerator-config ethos-u55-256 \
        --optimise Performance
        $ mv yolo-fastest_int8_vela.tflite yolo-fastest_int8_ethos-u55-256_opt-speed.tflite

     .. note:: The renamed model file is just how
        `yolo-fastest_int8_ethos-u55-256_opt-speed.tflite`_
        in this example directory is generated.

     .. _yolo-fastest_int8_ethos-u55-256_opt-speed.tflite: src/Model/yolo-fastest_int8_ethos-u55-256_opt-speed.tflite

More build options
===================

This section lists more build options.

Choosing object detection input
-------------------------------

This example supports the following choices for object detection input:

- ``CONFIG_NVT_ML_OD_INPUT_SENSOR_HM1055``

  Use CMOS image sensor HM1055 as object detection input

  This configuration option depends on the ``himax_hm1055`` shield (``--shield himax_hm1055``)
  and is automatically enabled with it.

- ``CONFIG_NVT_ML_OD_INPUT_IMAGE_BLOB``

  Use embedded image blob as object detection input (default)

Choosing object detection model
-------------------------------

This example supports the following choices for object detection model:

- ``CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8``

  Use YOLO-Fastest object detection model, type INT8,
  `no vela-compile`__

  .. __: `YOLO-Fastest v1.1`_

  .. note:: The model specified by this configuration needs TFLM tensor arena size 9.5 MB.
     With on-board HyperRAM only 8 MB, it would be neither build-able nor run-able.

- ``CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8_ETHOS_U55_256_SIZE``

  Use YOLO-Fastest object detection model, type INT8,
  vela-compile for Ethos-U55/256 MAC, `optimize for size`__ (default)

  .. __: `Optimizing for size`_

- ``CONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8_ETHOS_U55_256_SPEED``

  Use YOLO-Fastest object detection model, type INT8,
  vela-compile for Ethos-U55/256 MAC, `optimize for speed`__

  .. __: `Optimizing for speed`_

  .. note:: The model specified by this configuration needs TFLM tensor arena size 1.3 MB.
     HyperRAM must enable to accommodate the memory usage.

  .. code-block:: console

     $ west -v build \
     -b numaker_m55m1 \
     applications/NuMaker-Zephyr-TFLM-ObjectDetection \
     --shield winbond_hyperram \
     -- \
     -DCONFIG_NVT_ML_OD_MODEL_YOLO_FASTEST_INT8_ETHOS_U55_256_SPEED=y \
     -DCONFIG_NVT_ML_HYPERRAM_TENSOR_ARENA=y

Choosing object detection output display
----------------------------------------

Besides console, this example supports the following choices for
object detection output display:

- ``CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_FSA506``

  Use LCD FSA506 as object detection output over EBI

  This configuration option depends on the ``ids_fsa506`` shield (``--shield ids_fsa506``)
  and is automatically enabled with it.

- ``CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_ILI9341``

  Use LCD ILI9341 as object detection output over SPI

  This configuration option depends on the ``ilium_ili9341`` shield (``--shield ilium_ili9341``)
  and is automatically enabled with it.

- ``CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_LT7381``

  Use LCD LT7381 as object detection output over EBI

  This configuration option depends on the ``levetop_lt7381`` shield (``--shield levetop_lt7381``)
  and is automatically enabled with it.

- ``CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_NONE``

  No display as OD output (default)


Locating tensor arena at HyperRAM
---------------------------------

To locate TFLM tensor arena at HyperRAM, enable ``CONFIG_NVT_ML_HYPERRAM_TENSOR_ARENA``,
which depends on the ``winbond_hyperram`` shield (``--shield winbond_hyperram``).

.. code-block:: console

   $ west -v build \
   -b numaker_m55m1 \
   applications/NuMaker-Zephyr-TFLM-ObjectDetection \
   --shield winbond_hyperram \
   -- \
   -DCONFIG_NVT_ML_HYPERRAM_TENSOR_ARENA=y

Enabling profiling
------------------

To measure Ethos-U performance, enable ``CONFIG_NVT_ML_ETHOS_U_PROFILE``:
