/* Execute the pinned upstream game without its native learner or renderer. */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#ifdef GAME_2048
#include "g2048.h"
#else
#include "lightsout.h"
#endif

typedef struct Handle {
    Env env;
    obs_t observations[OBS_SIZE];
    float action, reward, terminal;
    int done, steps;
    float score;
} Handle;

void* game_create(unsigned int seed, const int* board, int horizon) {
    Handle* h = calloc(1, sizeof(Handle));
    if (!h) return NULL;
    h->env.rng = seed;
    h->env.agents[0].observations = h->observations;
    h->env.agents[0].actions = &h->action;
    h->env.agents[0].rewards = &h->reward;
    h->env.agents[0].terminals = &h->terminal;
    Dict kwargs = {0};
#ifdef GAME_2048
    (void)horizon;
    dict_set(&kwargs, "scaffolding_ratio", 0);
#else
    dict_set(&kwargs, "max_steps", horizon);
#endif
    puf_init(&h->env, &kwargs); puf_reset(&h->env);
    free(kwargs.items);
    if (board) {
#ifdef GAME_2048
        for (int i=0;i<16;i++) ((unsigned char*)h->env.grid)[i] = board[i];
        update_stats(&h->env); update_observations(&h->env);
#else
        h->env.lights_on = 0;
        for (int i=0;i<25;i++) { h->env.grid[i]=board[i]; h->env.lights_on+=board[i]; }
        compute_observations(&h->env);
#endif
    }
    return h;
}
void game_observe(void* pointer, int* board) {
    Handle* h=pointer;
    for (int i=0;i<OBS_SIZE;i++) board[i]=(int)h->observations[i];
}
float game_step(void* pointer, int action) {
    Handle* h=pointer;
    if (h->done || action<0 || action>=
#ifdef GAME_2048
        4
#else
        25
#endif
    ) return NAN;
    h->action=(float)action;
#ifdef GAME_2048
    float previous_logged_score=h->env.log.merge_score;
#endif
    puf_step(&h->env); h->steps++; h->done=(int)h->terminal;
#ifdef GAME_2048
    h->score=h->done ? h->env.log.merge_score-previous_logged_score : h->env.score;
#else
    h->score=h->env.log.perf;
#endif
    return h->reward;
}
int game_done(void* pointer) { return ((Handle*)pointer)->done; }
float game_score(void* pointer) { return ((Handle*)pointer)->score; }
void game_close(void* pointer) { Handle* h=pointer; if (h) { puf_close(&h->env); free(h); } }

#ifdef GAME_2048
/* Deterministic pre-spawn transition for independent-oracle comparison. */
int game_merge(const int* before, int action, int* after, float* reward, float* points) {
    Game game={0};
    if (action<0 || action>=4) return -1;
    for (int i=0;i<16;i++) ((unsigned char*)game.grid)[i]=before[i];
    *reward=0; *points=0;
    int changed=move(&game,action+1,reward,points);
    for (int i=0;i<16;i++) after[i]=((unsigned char*)game.grid)[i];
    return changed;
}
#endif
