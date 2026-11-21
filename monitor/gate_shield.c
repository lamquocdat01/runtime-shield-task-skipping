#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "gate_shield_types.h"
#include "gate_shield.h"

static bool valid_in_cpy;
static bool pi_in_cpy;
static bool rst_in_cpy;
static uint32_t s0[(1)] = {((uint32_t)(5))};
static size_t s0_idx = (0);

static uint32_t s0_get(size_t x) {
  return (s0)[((s0_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s0_gen(void) {
  return ((valid_in_cpy) && ((pi_in_cpy) || (((rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)))) >= ((uint32_t)(4))))) ? (uint32_t)(0) : ((((rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)))) >= ((uint32_t)(5))) ? (uint32_t)(5) : (((rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)))) + ((uint32_t)(1))));
}

static bool shield_out_0_guard(void) {
  return true;
}

static bool shield_out_0_arg0(void) {
  return (valid_in_cpy) && ((pi_in_cpy) || (((rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)))) >= ((uint32_t)(4))));
}

static bool shield_out_0_arg1(void) {
  return ((rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)))) >= ((uint32_t)(4));
}

static uint32_t shield_out_0_arg2(void) {
  return (rst_in_cpy) ? (uint32_t)(5) : ((s0_get)((0)));
}

void step(void) {
  uint32_t s0_tmp;
  bool shield_out_0_arg_temp0;
  bool shield_out_0_arg_temp1;
  uint32_t shield_out_0_arg_temp2;
  (valid_in_cpy) = (valid_in);
  (pi_in_cpy) = (pi_in);
  (rst_in_cpy) = (rst_in);
  if ((shield_out_0_guard)()) {
    {(shield_out_0_arg_temp0) = ((shield_out_0_arg0)());
     (shield_out_0_arg_temp1) = ((shield_out_0_arg1)());
     (shield_out_0_arg_temp2) = ((shield_out_0_arg2)());
     (shield_out)((shield_out_0_arg_temp0), (shield_out_0_arg_temp1), (shield_out_0_arg_temp2));}
  };
  (s0_tmp) = ((s0_gen)());
  ((s0)[s0_idx]) = (s0_tmp);
  (s0_idx) = (((s0_idx) + ((size_t)(1))) % ((size_t)(1)));
}
