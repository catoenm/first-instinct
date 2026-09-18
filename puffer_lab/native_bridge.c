#include "reservation_core.h"

void* reservation_create(int world, int horizon, const int* fees, const int* prior) {
    if (world < 0 || world >= 6 || horizon < 1 || horizon > 8) return NULL;
    int total = 0;
    for (int i=0; i<6; i++) { if (prior[i] < 0) return NULL; total += prior[i]; }
    for (int i=0; i<RES_ACTIONS; i++) if (fees[i] < 0) return NULL;
    if (!total) return NULL;
    ResState* s = malloc(sizeof(ResState));
    if (s) res_init(s,world,horizon,fees,prior);
    return s;
}
float reservation_step(void* ptr, int action) { return res_step(ptr,action); }
void reservation_observe(void* ptr, float* obs) { res_observe(ptr,obs); }
void reservation_dump(void* ptr, int* out) { res_dump(ptr,out); }
void reservation_destroy(void* ptr) { free(ptr); }
