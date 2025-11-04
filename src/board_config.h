/**************************************************************************//**
 * @file     board_config.h
 * @version  V1.00
 * @brief    Specify board component
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __BOARD_CONFIG_H__
#define __BOARD_CONFIG_H__

#include "NuMicro.h"

#if defined(CONFIG_BOARD_NUMAKER_M55M1)
#define __NUMAKER_M55M1__
//#define __NUMAKER_EZAI__
#endif

#if defined(CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_FSA506) || \
    defined(CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_LT7381)
    #define __EBI_LCD_PANEL__
    #define CONFIG_LCD_EBI                  EBI_BANK0
    #define CONFIG_LCD_EBI_ADDR             (EBI_BANK0_BASE_ADDR+(CONFIG_LCD_EBI*EBI_MAX_SIZE))
    #define CONFIG_LCD_EBI_CLK_MODULE       EBI0_MODULE
#if defined(CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_LT7381)
    #define LT7381_LCD_PANEL
#elif defined(CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_FSA506)
    #define FSA506_LCD_PANEL
#endif

#endif

#if defined(CONFIG_NVT_ML_OD_OUTPUT_DISPLAY_LCD_ILI9341)
    #define __SPI_LCD_PANEL__
    #define CONFIG_LCD_SPI_PORT             SPI2
    #define CONFIG_LCD_SPI_CLK_MODULE       SPI2_MODULE
    #define CONFIG_LCD_SPI_CLK_SEL          CLK_SPISEL_SPI2SEL_PCLK0

#endif

#endif
