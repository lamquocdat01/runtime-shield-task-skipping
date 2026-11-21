/* shield_glue.c -- topic 06, P2 Viec 3 (items 8-9): thin, UNVERIFIED glue around the
 * copilot-verifier-proved gate_shield.c / gate_ctl.c. It defines the extern input globals,
 * the trigger functions (they only store their arguments: no access to memory owned by the
 * generated code, as copilot-verifier requires of trigger implementations), and batch
 * entry points for ctypes / the overhead benchmark.
 * Build one shared object per Copilot program (each defines its own step()):
 *   gcc -O2 -fPIC -shared -Wl,-Bsymbolic -DSHIELD gate_shield.c shield_glue.c -o libshield.so
 *   gcc -O2 -fPIC -shared -Wl,-Bsymbolic -DCTL    gate_ctl.c    shield_glue.c -o libctl.so
 */
#include <stdbool.h>
#include <stdint.h>

#ifdef SHIELD
#include "gate_shield.h"
bool valid_in, pi_in, rst_in;
static bool o_e, o_force;
static uint32_t o_run;

void shield_out(bool e, bool force, uint32_t run) {
  o_e = e;
  o_force = force;
  o_run = run;
}

/* one frame; returns bit0 = OPEN after the shield, bit1 = forced by the shield */
int shield_frame(int pi, int valid, int rst) {
  pi_in = pi != 0;
  valid_in = valid != 0;
  rst_in = rst != 0;
  step();
  return (o_e ? 1 : 0) | (o_force ? 2 : 0);
}

/* open loop over arrays of n frames */
long shield_batch(const uint8_t *pi, const uint8_t *valid, const uint8_t *rst, uint8_t *out,
                  long n) {
  long opens = 0;
  for (long t = 0; t < n; t++) {
    pi_in = pi[t] != 0;
    valid_in = valid[t] != 0;
    rst_in = rst[t] != 0;
    step();
    out[t] = o_e ? 1 : 0;
    opens += out[t];
  }
  return opens;
}
#endif

#ifdef CTL
#include "gate_ctl.h"
bool g_in, valid_in, rst_in;
static bool o_ge, o_gt, o_gtl;

void ctl_out(bool ge, bool gt, bool gtl, uint32_t fge, uint32_t fgt) {
  (void)fge;
  (void)fgt;
  o_ge = ge;
  o_gt = gt;
  o_gtl = gtl;
}

/* out[t] = bit0 o_ge | bit1 o_gt | bit2 o_gtL */
void ctl_batch(const uint8_t *g, const uint8_t *valid, const uint8_t *rst, uint8_t *out, long n) {
  for (long t = 0; t < n; t++) {
    g_in = g[t] != 0;
    valid_in = valid[t] != 0;
    rst_in = rst[t] != 0;
    step();
    out[t] = (uint8_t)((o_ge ? 1 : 0) | (o_gt ? 2 : 0) | (o_gtl ? 4 : 0));
  }
}
#endif
