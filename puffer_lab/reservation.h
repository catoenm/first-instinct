/* PufferLib 5.0 environment adapter. The same core has a separate SQL audit. */
typedef float obs_t;
#include "pufferenv.h"
#include "reservation_core.h"

#define ACT_SIZES {RES_ACTIONS}
#define OBS_SIZE RES_OBS
#define NUM_ATNS 1

struct Log { float perf, score, n; };
struct Env {
    Log log;
    int num_agents;
    unsigned int rng;
    Agent agents[1];
    int tag, boundary_reached;
    ResState state;
    int fees[9], prior[6], horizon, fixed_world;
    float episode_return;
};

static const int RES_PROFILE_FEES[4][9] = {
    {1,1,16,4,12,12,12,8,0}, {8,8,4,4,12,12,12,8,0},
    {1,1,16,4,12,12,12,8,0}, {1,1,16,4,12,12,12,8,0}
};

void puf_init(Env* env, Dict* kwargs) {
    DictItem* item = dict_find(kwargs,"profile");
    int profile = item ? (int)item->value : 0;
    if (profile < 0 || profile > 3) { fprintf(stderr,"Invalid reservation profile\n");exit(1); }
    item = dict_find(kwargs,"world");
    env->fixed_world = item ? (int)item->value : -1;
    if (env->fixed_world < -1 || env->fixed_world > 5) { fprintf(stderr,"Invalid reservation world\n");exit(1); }
    env->horizon = profile == 2 ? 3 : 6;
    memcpy(env->fees,RES_PROFILE_FEES[profile],sizeof(env->fees));
    for (int i=0; i<6; i++) env->prior[i] = i < 4 || profile == 3 ? 1 : 0;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    /* All actions are available in every hidden world. */
    env->agents[0].action_mask = NULL;
}

void puf_reset(Env* env) {
    int world = env->fixed_world;
    if (world < 0) {
        int total = 0;
        for (int i=0; i<6; i++) total += env->prior[i];
        unsigned int draw = rand_r(&env->rng) % (unsigned int)total;
        for (world=0; world<5 && draw >= (unsigned int)env->prior[world]; world++) draw -= env->prior[world];
    }
    res_init(&env->state,world,env->horizon,env->fees,env->prior);
    env->episode_return = 0;
    res_observe(&env->state,env->agents[0].observations);
}

void puf_step(Env* env) {
    int action = (int)env->agents[0].actions[0];
    if (action < 0 || action >= RES_ACTIONS) { fprintf(stderr,"Invalid reservation action\n");exit(1); }
    float reward = res_step(&env->state,action);
    int terminal = env->state.done;
    env->episode_return += reward;
    if (terminal) {
        env->log.perf += env->state.outcome == 1;
        env->log.score += env->episode_return;
        env->log.n += 1;
        puf_reset(env);
    } else res_observe(&env->state,env->agents[0].observations);
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = (float)terminal;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out,"perf",log->perf);
    dict_set(out,"score",log->score);
    dict_set(out,"n",log->n);
}
void puf_render(Env* env) { (void)env; }
void puf_close(Env* env) { (void)env; }
