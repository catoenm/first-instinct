/* A finite transactional state machine; SQL execution is checked independently. */
#ifndef FIRST_INSTINCT_RESERVATION_CORE_H
#define FIRST_INSTINCT_RESERVATION_CORE_H
#include <stdlib.h>
#include <string.h>

#define RES_ACTIONS 9
#define RES_OBS 39
#define RES_FIELDS 20

typedef struct {
    int a, b, account, header, alloc_a, alloc_b, added_a, added_b;
    int initial_a, initial_b, initial_account, step, horizon, done, outcome;
    int cost, code, known_a, known_b, known_account;
    int fees[RES_ACTIONS], prior[6];
} ResState;

static const int RES_WORLDS[6][3] = {
    {1,1,1}, {0,2,1}, {2,0,1}, {1,1,0}, {0,2,0}, {2,0,0}
};

static void res_init(ResState* s, int world, int horizon, const int* fees, const int* prior) {
    memset(s, 0, sizeof(*s));
    s->a = s->initial_a = RES_WORLDS[world][0];
    s->b = s->initial_b = RES_WORLDS[world][1];
    s->account = s->initial_account = RES_WORLDS[world][2];
    s->horizon = horizon;
    s->known_a = s->known_b = s->known_account = -1;
    memcpy(s->fees, fees, RES_ACTIONS*sizeof(int));
    memcpy(s->prior, prior, 6*sizeof(int));
}

static int res_verify(const ResState* s) {
    if (s->a < 0 || s->b < 0 || s->a+s->alloc_a != s->initial_a+s->added_a ||
        s->b+s->alloc_b != s->initial_b+s->added_b ||
        ((s->alloc_a || s->alloc_b) && !s->header) ||
        (s->header && !s->account)) return 4;
    if (s->header && s->alloc_a == 1 && s->alloc_b == 1) return 1;
    if (s->header || s->alloc_a || s->alloc_b) return 3;
    return 2;
}

static void res_reserve(ResState* s, int atomic) {
    ResState before = *s;
    int code;
    /* SQLite checks the request primary key before its foreign key here. */
    if (s->header) { s->code = 7; return; }
    if (!s->account) { s->known_account = 0; s->code = 4; return; }
    s->header = 1;
    s->known_account = 1;
    if (s->a == 0) { code = 5; goto error; }
    s->a--;
    s->alloc_a = 1;
    if (s->known_a >= 0) s->known_a--;
    if (s->b == 0) { code = 6; goto error; }
    s->b--;
    s->alloc_b = 1;
    if (s->known_b >= 0) s->known_b--;
    s->code = 3;
    return;
error:
    if (atomic) *s = before;
    s->known_account = 1;
    if (code == 5) s->known_a = 0;
    if (code == 6) s->known_b = 0;
    s->code = code;
}

static float res_step(ResState* s, int action) {
    if (s->done || action < 0 || action >= RES_ACTIONS) return 0;
    s->step++;
    s->cost += s->fees[action];
    int fee = s->fees[action];
    switch (action) {
        case 0: s->known_a = s->a; s->known_b = s->b; s->code = 1; break;
        case 1: s->known_account = s->account; s->code = 2; break;
        case 2: res_reserve(s, 1); break;
        case 3: res_reserve(s, 0); break;
        case 4:
            s->code = 12;
            if (s->a == 0) { s->a++; s->added_a++; s->known_a = 1; s->code = 8; }
            break;
        case 5:
            s->code = 12;
            if (s->b == 0) { s->b++; s->added_b++; s->known_b = 1; s->code = 8; }
            break;
        case 6:
            s->code = s->account ? 12 : 9;
            s->account = s->known_account = 1;
            break;
        case 7:
            s->a += s->alloc_a; s->b += s->alloc_b;
            if (s->known_a >= 0) s->known_a += s->alloc_a;
            if (s->known_b >= 0) s->known_b += s->alloc_b;
            s->header = s->alloc_a = s->alloc_b = 0; s->code = 10;
            break;
        case 8: s->code = 11; break;
    }
    int terminal = 0;
    if (action == 8 || s->step >= s->horizon) {
        s->done = 1; s->outcome = res_verify(s);
        terminal = s->outcome == 1 ? 400 : s->outcome >= 3 ? -400 : 0;
    }
    return (terminal-fee)/400.0f;
}

static void res_observe(const ResState* s, float* obs) {
    int i = 0;
    obs[i++] = s->step/8.0f; obs[i++] = (s->horizon-s->step)/8.0f;
    obs[i++] = s->known_a < 0 ? -1.0f : s->known_a/4.0f;
    obs[i++] = s->known_b < 0 ? -1.0f : s->known_b/4.0f;
    obs[i++] = (float)s->known_account;
    obs[i++] = (float)s->header; obs[i++] = (float)s->alloc_a; obs[i++] = (float)s->alloc_b;
    obs[i++] = s->cost/400.0f;
    for (int k=0; k<13; k++) obs[i++] = s->code == k ? 1.0f : 0.0f;
    for (int k=0; k<RES_ACTIONS; k++) obs[i++] = s->fees[k]/400.0f;
    int total = 0;
    for (int k=0; k<6; k++) total += s->prior[k];
    for (int k=0; k<6; k++) obs[i++] = s->prior[k]/(float)total;
    obs[i++] = s->horizon/8.0f; obs[i++] = 1.0f;
}

static void res_dump(const ResState* s, int* out) {
    const int values[RES_FIELDS] = {s->a,s->b,s->account,s->header,s->alloc_a,s->alloc_b,
        s->added_a,s->added_b,s->initial_a,s->initial_b,s->initial_account,s->step,
        s->horizon,s->done,s->outcome,s->cost,s->code,s->known_a,s->known_b,s->known_account};
    memcpy(out, values, sizeof(values));
}
#endif
