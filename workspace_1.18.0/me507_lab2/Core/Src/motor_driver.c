/*
 * motor_driver.c
 *
 *  Created on: Apr 17, 2025
 *      Author: kyle
 */

#include "motor_driver.h"

void motor_set(motor* m, uint32_t duty_cycle, int8_t direction) {
	if (duty_cycle > 100) duty_cycle = 100;

	m->duty_cycle = duty_cycle;
	m->direction = direction;

	uint32_t autoreload = __HAL_TIM_GET_AUTORELOAD(m->htim);
	uint32_t pulse = autoreload - ((autoreload + 1) * m->duty_cycle/100);

	if (m->direction > 0) {
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel1, autoreload);
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel2, pulse);
	} else if (m->direction < 0) {
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel1, pulse);
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel2, autoreload);
	}

}

void motor_enable(motor* m) {
	uint32_t autoreload = __HAL_TIM_GET_AUTORELOAD(m->htim);
	uint32_t pulse = autoreload - ((autoreload + 1) * m->duty_cycle/100);

	if (m->direction > 0) {
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel1, autoreload);
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel2, pulse);
	} else {
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel1, pulse);
		__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel2, autoreload);
	}

}

void motor_disable(motor* m) {
	uint32_t autoreload = __HAL_TIM_GET_AUTORELOAD(m->htim);
	__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel1, autoreload);
	__HAL_TIM_SET_COMPARE(m->htim, m->pwm_channel2, autoreload);
}
