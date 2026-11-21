#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "gate_ctl_types.h"
#include "gate_ctl.h"

static bool rst_in_cpy;
static bool valid_in_cpy;
static bool g_in_cpy;
static uint32_t s0[(1)] = {((uint32_t)(0))};
static uint32_t s1[(1)] = {((uint32_t)(5))};
static uint32_t s2[(1)] = {((uint32_t)(5))};
static uint32_t s4[(1)] = {((uint32_t)(5))};
static uint32_t s3[(1)] = {((uint32_t)(5))};
static size_t s0_idx = (0);
static size_t s1_idx = (0);
static size_t s2_idx = (0);
static size_t s4_idx = (0);
static size_t s3_idx = (0);

static uint32_t s0_get(size_t x) {
  return (s0)[((s0_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s1_get(size_t x) {
  return (s1)[((s1_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s2_get(size_t x) {
  return (s2)[((s2_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s4_get(size_t x) {
  return (s4)[((s4_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s3_get(size_t x) {
  return (s3)[((s3_idx) + (x)) % ((size_t)(1))];
}

static uint32_t s0_gen(void) {
  return (rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))));
}

static uint32_t s1_gen(void) {
  return ((valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s1_get)((0)))) >= ((uint32_t)(4)))))) ? (uint32_t)(0) : ((((rst_in_cpy) ? (uint32_t)(5) : ((s1_get)((0)))) >= ((uint32_t)(5))) ? (uint32_t)(5) : (((rst_in_cpy) ? (uint32_t)(5) : ((s1_get)((0)))) + ((uint32_t)(1))));
}

static uint32_t s2_gen(void) {
  return ((valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s2_get)((0)))) > ((uint32_t)(4)))))) ? (uint32_t)(0) : ((((rst_in_cpy) ? (uint32_t)(5) : ((s2_get)((0)))) >= ((uint32_t)(5))) ? (uint32_t)(5) : (((rst_in_cpy) ? (uint32_t)(5) : ((s2_get)((0)))) + ((uint32_t)(1))));
}

static uint32_t s4_gen(void) {
  return ((valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || ((((rst_in_cpy) ? (uint32_t)(5) : ((s3_get)((0)))) > ((uint32_t)(4))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s4_get)((0)))) >= ((uint32_t)(4))))))) ? (uint32_t)(0) : ((((rst_in_cpy) ? (uint32_t)(5) : ((s4_get)((0)))) >= ((uint32_t)(5))) ? (uint32_t)(5) : (((rst_in_cpy) ? (uint32_t)(5) : ((s4_get)((0)))) + ((uint32_t)(1))));
}

static uint32_t s3_gen(void) {
  return ((valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || ((((rst_in_cpy) ? (uint32_t)(5) : ((s3_get)((0)))) > ((uint32_t)(4))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s4_get)((0)))) >= ((uint32_t)(4))))))) ? (uint32_t)(0) : ((((rst_in_cpy) ? (uint32_t)(5) : ((s3_get)((0)))) >= ((uint32_t)(5))) ? (uint32_t)(5) : (((rst_in_cpy) ? (uint32_t)(5) : ((s3_get)((0)))) + ((uint32_t)(1))));
}

static bool ctl_out_0_guard(void) {
  return true;
}

static bool ctl_out_0_arg0(void) {
  return (valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s1_get)((0)))) >= ((uint32_t)(4)))));
}

static bool ctl_out_0_arg1(void) {
  return (valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s2_get)((0)))) > ((uint32_t)(4)))));
}

static bool ctl_out_0_arg2(void) {
  return (valid_in_cpy) && ((g_in_cpy) || (((!(rst_in_cpy)) && (((rst_in_cpy) ? (uint32_t)(0) : (((((s0_get)((0))) + ((uint32_t)(1))) >= ((uint32_t)(15))) ? (uint32_t)(0) : (((s0_get)((0))) + ((uint32_t)(1))))) == ((uint32_t)(0)))) || ((((rst_in_cpy) ? (uint32_t)(5) : ((s3_get)((0)))) > ((uint32_t)(4))) || (((rst_in_cpy) ? (uint32_t)(5) : ((s4_get)((0)))) >= ((uint32_t)(4))))));
}

static uint32_t ctl_out_0_arg3(void) {
  return (rst_in_cpy) ? (uint32_t)(5) : ((s1_get)((0)));
}

static uint32_t ctl_out_0_arg4(void) {
  return (rst_in_cpy) ? (uint32_t)(5) : ((s2_get)((0)));
}

void step(void) {
  uint32_t s0_tmp;
  uint32_t s1_tmp;
  uint32_t s2_tmp;
  uint32_t s4_tmp;
  uint32_t s3_tmp;
  bool ctl_out_0_arg_temp0;
  bool ctl_out_0_arg_temp1;
  bool ctl_out_0_arg_temp2;
  uint32_t ctl_out_0_arg_temp3;
  uint32_t ctl_out_0_arg_temp4;
  (rst_in_cpy) = (rst_in);
  (valid_in_cpy) = (valid_in);
  (g_in_cpy) = (g_in);
  if ((ctl_out_0_guard)()) {
    {(ctl_out_0_arg_temp0) = ((ctl_out_0_arg0)());
     (ctl_out_0_arg_temp1) = ((ctl_out_0_arg1)());
     (ctl_out_0_arg_temp2) = ((ctl_out_0_arg2)());
     (ctl_out_0_arg_temp3) = ((ctl_out_0_arg3)());
     (ctl_out_0_arg_temp4) = ((ctl_out_0_arg4)());
     (ctl_out)((ctl_out_0_arg_temp0), (ctl_out_0_arg_temp1), (ctl_out_0_arg_temp2), (ctl_out_0_arg_temp3), (ctl_out_0_arg_temp4));}
  };
  (s0_tmp) = ((s0_gen)());
  (s1_tmp) = ((s1_gen)());
  (s2_tmp) = ((s2_gen)());
  (s4_tmp) = ((s4_gen)());
  (s3_tmp) = ((s3_gen)());
  ((s0)[s0_idx]) = (s0_tmp);
  ((s1)[s1_idx]) = (s1_tmp);
  ((s2)[s2_idx]) = (s2_tmp);
  ((s4)[s4_idx]) = (s4_tmp);
  ((s3)[s3_idx]) = (s3_tmp);
  (s0_idx) = (((s0_idx) + ((size_t)(1))) % ((size_t)(1)));
  (s1_idx) = (((s1_idx) + ((size_t)(1))) % ((size_t)(1)));
  (s2_idx) = (((s2_idx) + ((size_t)(1))) % ((size_t)(1)));
  (s4_idx) = (((s4_idx) + ((size_t)(1))) % ((size_t)(1)));
  (s3_idx) = (((s3_idx) + ((size_t)(1))) % ((size_t)(1)));
}
