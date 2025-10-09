/**************************************************************************//**
 * @file     BoardInit.cpp
 * @version  V1.00
 * @brief    Target board initiate function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <cstdio>

#include "NuMicro.h"
#include "log_macros.h"

#include "ethosu_npu_init.h"
#if defined(__ZEPHYR__)
#if defined(CONFIG_NVT_ML_HYPERRAM)
#include "hyperram_code.h"
#endif
#endif

#define DESIGN_NAME "M55M1"
#if defined(__ZEPHYR__)
#if defined(CONFIG_NVT_ML_HYPERRAM)
#define HYPERRAM_SPIM_PORT SPIM0
#endif
#endif

static void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* On zephyr, go in zephyr way */
#if defined(__ZEPHYR__)
    /* Enable FMC0 module clock to keep FMC clock when CPU idle but NPU running*/
    CLK_EnableModuleClock(FMC0_MODULE);

#if defined(CONFIG_NVT_ML_OD_INPUT_CCAP)
    /* Enable CCAP0 module clock */
    CLK_EnableModuleClock(CCAP0_MODULE);
#endif

#if defined(CONFIG_NVT_ML_HYPERRAM)
    HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);
#endif
#else
    /* Enable Internal RC 12MHz clock */
    CLK_EnableXtalRC(CLK_SRCCTL_HIRCEN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);

    /* Enable HXT clock */
    CLK_EnableXtalRC(CLK_SRCCTL_HXTEN_Msk);

    /* Waiting for HXT clock ready */
    CLK_WaitClockReady(CLK_STATUS_HXTSTB_Msk);

    /* Switch SCLK clock source to APLL0 and Enable APLL0 220MHz clock */
    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HIRC, FREQ_220MHZ);

    /* Update System Core Clock */
    /* User can use SystemCoreClockUpdate() to calculate SystemCoreClock. */
    SystemCoreClockUpdate();

    /* Enable GPIO module clock */
    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

    /* Enable FMC0 module clock to keep FMC clock when CPU idle but NPU running*/
    CLK_EnableModuleClock(FMC0_MODULE);

    /* Enable NPU module clock */
    CLK_EnableModuleClock(NPU0_MODULE);

    /* Enable CCAP0 module clock */
    CLK_EnableModuleClock(CCAP0_MODULE);

    /* Select UART6 module clock source as HIRC and UART6 module clock divider as 1 */
    SetDebugUartCLK();

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Set multi-function pins for UART RXD and TXD */
    SetDebugUartMFP();

    HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);
#endif
}

/**
  * @brief Initiate the hardware resources of board
  * @return 0: Success, <0: Fail
  * @details Initiate clock, UART, NPU, hyperflash/hyperRAM
  * \hideinitializer
  */
int BoardInit(void)
{
    /* Unlock protected registers */
    SYS_UnlockReg();

    SYS_Init();

    /* On zephyr, go in zephyr way */
#if !defined(__ZEPHYR__)
    /* UART init - will enable valid use of printf (stdout
     * re-directed at this UART (UART6) */
    InitDebugUart();
#endif

    SYS_LockReg();                   /* Unlock register lock protect */

#if defined(__ZEPHYR__)
#if defined(CONFIG_NVT_ML_HYPERRAM)
    HyperRAM_Init(HYPERRAM_SPIM_PORT);
    /* Enter direct-mapped mode to run new applications */
    SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);
#endif
#endif

    info("%s: complete\n", __FUNCTION__);

#if defined(ARM_NPU)

    int state;

    /* If Arm Ethos-U NPU is to be used, we initialise it here */
    if (0 != (state = arm_ethosu_npu_init()))
    {
        return state;
    }

#endif /* ARM_NPU */

    /* Print target design info */
    info("Target system: %s\n", DESIGN_NAME);

    return 0;
}


