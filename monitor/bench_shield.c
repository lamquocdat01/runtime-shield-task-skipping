/* bench_shield.c -- topic 06, P2 Viec 3 (item 9): per-frame cost of the verified lazy shield
 * (gate_shield.c, -O2) replaying the 121-video policy trace (166,589 frames).
 * Usage: bench_shield <pi.u8> <valid.u8> <rst.u8> <runs>   (raw uint8 files, one byte/frame)
 * Prints one line per run: run, total_ns, ns_per_frame, opens (checksum vs dead-code removal).
 * Build: gcc -O2 -DSHIELD gate_shield.c shield_glue.c bench_shield.c -o bench_shield
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long shield_batch(const uint8_t *pi, const uint8_t *valid, const uint8_t *rst, uint8_t *out,
                  long n);

static uint8_t *load(const char *p, long *n) {
  FILE *f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END);
  *n = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc((size_t)*n);
  if (fread(b, 1, (size_t)*n, f) != (size_t)*n) { perror("read"); exit(1); }
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage\n"); return 2; }
  long n, n2, n3;
  uint8_t *pi = load(argv[1], &n), *va = load(argv[2], &n2), *rs = load(argv[3], &n3);
  if (n != n2 || n != n3) { fprintf(stderr, "length mismatch\n"); return 1; }
  int runs = atoi(argv[4]);
  uint8_t *out = malloc((size_t)n);
  shield_batch(pi, va, rs, out, n); /* warm-up */
  for (int r = 0; r < runs; r++) {
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    long opens = shield_batch(pi, va, rs, out, n);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double ns = (b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec);
    printf("%d %.0f %.3f %ld\n", r, ns, ns / n, opens);
  }
  return 0;
}
