/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body for CoreXY stepper motor control system
  * @details        : Implements dual UART communication (PC and Raspberry Pi)
  *                   with CoreXY motion control, position commands processing,
  *                   and stepper motor coordination with limit switches.
  * @author         : Kyle Schumacher
  * @date           : Jun 9, 2025
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f4xx_hal.h"
#include "stepper_driver.h"
#include "stepper_manager.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
StepperMotor motor1;    ///< Motor A for CoreXY system (X+Y movement)
StepperMotor motor2;    ///< Motor B for CoreXY system (X-Y movement)
StepperManager mgr;     ///< High-level motion controller for coordinated moves
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim2;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint32_t PWM_duty_cycle = 4800;    ///< PWM duty cycle (not currently used)
uint32_t counterValue = 0;         ///< General purpose counter
uint32_t board_dim_x = 285.75;     ///< Physical board dimension X in mm
uint32_t board_dim_y = 487.3;      ///< Physical board dimension Y in mm

#define RX_BUFFER_SIZE 64          ///< UART receive buffer size in bytes
uint8_t uart1_rx_data, uart2_rx_data;  ///< Single byte buffers for UART interrupts
uint8_t uart_rx_data;              ///< Common variable for processing received data
HAL_StatusTypeDef uart_status;     ///< UART operation status
uint8_t uart_rx_buffer[RX_BUFFER_SIZE];  ///< Command assembly buffer
uint8_t uart_rx_index = 0;         ///< Current position in receive buffer
bool new_data = false;             ///< Flag indicating new UART data received

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM2_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
char tx_buff[64];           ///< Transmit buffer for UART responses
float max_speed = 4000.0f;  ///< Maximum motor speed in steps/second
float max_accel = 20000.0f; ///< Maximum motor acceleration in steps/second²
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_USART2_UART_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
  StepperMotor_Init(
	  &motor1,
	  GPIOB, GPIO_PIN_8,
	  GPIOB, GPIO_PIN_6,
	  max_speed,
	  max_accel
  );

  StepperMotor_Init(
	  &motor2,
	  GPIOB, GPIO_PIN_9,
	  GPIOB, GPIO_PIN_7,
	  max_speed,
	  max_accel
  );

  float steps_per_mm = 80.0f;
  StepperManager_Init(
		  &mgr,
		  &motor1,
		  &motor2,
		  GPIOB, GPIO_PIN_10, // X limit switch
		  GPIOB, GPIO_PIN_12,	// Y limit switch
		  steps_per_mm,
		  70.0f, // X min soft stop
		  215.0f,	// X max soft stop
		  80.0f,	// Y min soft stop
		  270.0f,	// Y max soft stop
		  5.0f	// Backoff in mm after limit switch
  );

  HAL_TIM_Base_Start_IT(&htim2);

  // TEST
  // Assumes PB6 is already GPIO_Output (no timer ISR involved here).
  // We'll manually pulse STEP at a slow rate so you can see the waveform.

//  {
//      int32_t start_pos_1 = motor1.current_position;
//      int32_t target_pos_1 = start_pos_1 + 1000;
//
//      // Begin nonblocking move:
//      StepperMotor_MoveTo(&motor1, target_pos_1);
//
//      // Wait until motor1 finishes:
//      while (StepperMotor_IsMoving(&motor1)) {
//          // You can put a small delay or toggle an LED here if you like, e.g. HAL_Delay(1).
//          // But since the stepper ISR is in TIM2, we just spin-wait.
//      }
//
//      HAL_Delay(200); // 200 ms pause
//
//      // Move it back to start:
//      StepperMotor_MoveTo(&motor1, start_pos_1);
//      while (StepperMotor_IsMoving(&motor1)) {
//          // spin-wait
//      }
//
//      HAL_Delay(200);
//
//      int32_t start_pos_2 = motor2.current_position;
//      int32_t target_pos_2 = start_pos_2 + 1000;
//
//      // Begin nonblocking move:
//      StepperMotor_MoveTo(&motor2, target_pos_2);
//
//      // Wait until motor1 finishes:
//      while (StepperMotor_IsMoving(&motor2)) {
//          // You can put a small delay or toggle an LED here if you like, e.g. HAL_Delay(1).
//          // But since the stepper ISR is in TIM2, we just spin-wait.
//      }
//
//      HAL_Delay(200); // 200 ms pause
//
//      // Move it back to start:
//      StepperMotor_MoveTo(&motor2, start_pos_2);
//      while (StepperMotor_IsMoving(&motor2)) {
//          // spin-wait
//      }
//
//      HAL_Delay(200);
//  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  HAL_UART_Receive_IT(&huart1, &uart1_rx_data, 1);
  HAL_UART_Receive_IT(&huart2, &uart2_rx_data, 1);
  StepperManager_StartCalibration(&mgr);
  char msg[] = "Hello from STM32!\r\n";
  HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);

  while (1)
  {
	  StepperManager_Update(&mgr);

	  if (new_data == true) {
		  new_data = false;
		  if (uart_rx_data == '\r' || uart_rx_data == '\n') {
			  uart_rx_buffer[uart_rx_index] = '\0';

			  char reply[64];
			  int len = 0;
			  if (uart_rx_buffer[0] == 'M' && uart_rx_index >= 3) {
				  int raw_pos_x = four_ascii_to_int(uart_rx_buffer[1], uart_rx_buffer[2], uart_rx_buffer[3], uart_rx_buffer[4]);
				  int raw_pos_y = four_ascii_to_int(uart_rx_buffer[5], uart_rx_buffer[6], uart_rx_buffer[7], uart_rx_buffer[8]);
				  float pos_x = raw_pos_x / 10.0f;
				  float pos_y = raw_pos_y / 10.0f;

				  // Process movement command - no need to check if IDLE now
				  StepperManager_MoveTo(&mgr, pos_x, pos_y);
				  
				  // Include current state and position in response for debugging
				  char* state_str = "UNKNOWN";
				  if (mgr.state == MANAGER_IDLE) state_str = "IDLE";
				  else if (mgr.state == MANAGER_MOVING) state_str = "MOVING";
				  else if (mgr.state == MANAGER_CALIBRATING) state_str = "CALIBRATING";
				  
				  len = snprintf(reply, sizeof(reply), "MoveTo: X=%0.1f Y=%0.1f State=%s\r\n", 
								pos_x, pos_y, state_str);
			  } else {
				  len = snprintf(reply, sizeof(reply), "Invalid cmd format\r\n");
			  }

			  HAL_UART_Transmit(&huart1, (uint8_t*)reply, len, 1000);

			  // Echo input for debugging
			  for (uint16_t i = 0; i < uart_rx_index; ++i) {
				  HAL_UART_Transmit(&huart1,
									(uint8_t*)&uart_rx_buffer[i],
									1,
									1000);
			  }
			  const char *nl = "\r\n";
			  HAL_UART_Transmit(&huart1,
								(uint8_t*)nl,
								2,
								100);

			  uart_rx_index = 0;
		  }
		  else {
			  if (uart_rx_index < RX_BUFFER_SIZE - 1) {
				  uart_rx_buffer[uart_rx_index++] = uart_rx_data;
			  }
		  }
	  }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 2499;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9, GPIO_PIN_RESET);

  /*Configure GPIO pin : PC13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PB10 PB12 */
  GPIO_InitStruct.Pin = GPIO_PIN_10|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PB6 PB7 PB8 PB9 */
  GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/**
 * @brief Timer interrupt callback for stepper motor updates
 * @param htim Pointer to timer handle that triggered the interrupt
 * @details Called automatically by HAL when TIM2 period elapses.
 *          Updates both stepper motors to generate step pulses and
 *          maintain motion profiles. This function runs at high frequency
 *          and should be kept minimal.
 * @note Timer frequency determines maximum step rate and motion smoothness
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
	if (htim->Instance == TIM2)
	{
		StepperMotor_Update(&motor1);
		StepperMotor_Update(&motor2);
	}
}

/**
 * @brief UART receive complete interrupt callback
 * @param huart Pointer to UART handle that received data
 * @details Handles incoming data from both UART1 (PC/PuTTY) and UART2 (Raspberry Pi).
 *          Provides echo functionality for PC connection and visual feedback for
 *          Raspberry Pi communication via LED toggle. Automatically re-enables
 *          UART interrupts for continuous reception.
 * @note UART1 data is echoed back to sender for debugging
 * @note UART2 data toggles onboard LED to indicate activity
 * @note Both UARTs share the same processing pipeline via uart_rx_data
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART1) {
		// Data from UART1 (PC/PuTTY)
		new_data = true;
		uart_rx_data = uart1_rx_data;
		HAL_UART_Transmit(&huart1, (uint8_t*)&uart1_rx_data, 1, 1000); // Echo back to UART1
		HAL_UART_Receive_IT(&huart1, &uart1_rx_data, 1); // Re-enable UART1 interrupt
	}
	else if (huart->Instance == USART2) {
		// Data from UART2 (Raspberry Pi)
		new_data = true;
		uart_rx_data = uart2_rx_data;
		HAL_UART_Transmit(&huart1, (uint8_t*)&uart2_rx_data, 1, 1000); // Echo to UART1 for debugging
		
		// Critical: Start a new receive operation BEFORE processing the data
		// This prevents missing bytes when they come in rapid succession
		HAL_UART_Receive_IT(&huart2, &uart2_rx_data, 1); // Re-enable UART2 interrupt
		
		// Debug marker to know UART2 is active
		HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13); // Toggle LED to show UART2 activity
	}
}

/**
 * @brief Convert single hexadecimal character to integer
 * @param c Hexadecimal character ('0'-'9', 'A'-'F', 'a'-'f')
 * @return Integer value (0-15) or undefined for invalid input
 * @note This function doesn't validate input - ensure valid hex chars
 * @warning No bounds checking performed on input
 */
int hex_to_int(uint8_t c) {
	if (c >= 97)
		c = c - 32;
	int first = c / 16 - 3;
	int second = c % 16;
	int result = first*10 + second;
	if (result > 9) result--;
	return result;
}

/**
 * @brief Convert four hexadecimal characters to integer
 * @param num1 First hex digit (most significant)
 * @param num2 Second hex digit
 * @param num3 Third hex digit  
 * @param num4 Fourth hex digit (least significant)
 * @return Combined integer value from four hex digits
 * @note Uses hex_to_int() internally for each digit conversion
 * @warning No validation performed on input characters
 */
int four_hex_to_int(uint8_t num1, uint8_t num2, uint8_t num3, uint8_t num4) {
	uint8_t int1 = hex_to_int(num1);
	uint8_t int2 = hex_to_int(num2);
	uint8_t int3 = hex_to_int(num3);
	uint8_t int4 = hex_to_int(num4);

	uint8_t combined = int1 * 1000 + int2 * 100 + int3 * 10 + int4;
	return combined;
}

/**
 * @brief Convert four ASCII decimal characters to integer
 * @param d1 First decimal digit (thousands place)
 * @param d2 Second decimal digit (hundreds place)
 * @param d3 Third decimal digit (tens place)
 * @param d4 Fourth decimal digit (ones place)
 * @return Combined integer value (0-9999) or -1 for invalid input
 * @details Validates each character is a valid decimal digit ('0'-'9')
 *          before performing conversion. Used for parsing position commands.
 * @note Returns -1 if any character is not a valid decimal digit
 * @note Result range is 0-9999 for valid inputs
 */
int four_ascii_to_int(uint8_t d1, uint8_t d2, uint8_t d3, uint8_t d4) {
	if (d1 < '0' || d1 > '9') return -1;
	if (d2 < '0' || d2 > '9') return -1;
	if (d3 < '0' || d3 > '9') return -1;
	if (d4 < '0' || d4 > '9') return -1;

	return (d1 - '0') * 1000 +
	       (d2 - '0') * 100 +
	       (d3 - '0') * 10 +
	       (d4 - '0') * 1;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
