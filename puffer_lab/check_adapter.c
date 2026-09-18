/* Headless lifecycle test against the actual pinned PufferLib header. */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include "reservation.h"

int main(void) {
    int transitions = 0, episodes = 0;
    for (int profile=0; profile<4; profile++) for (int world=0; world<6; world++) {
        Env env = {0}; env.rng = 101;
        Dict kwargs = {0};
        dict_set(&kwargs,"profile",profile);dict_set(&kwargs,"world",world);
        float observations[RES_OBS], actions[1], rewards[1], terminals[1];
        env.agents[0].observations = observations;env.agents[0].actions = actions;
        env.agents[0].rewards = rewards;env.agents[0].terminals = terminals;
        puf_init(&env,&kwargs);puf_reset(&env);
        ResState expected;
        res_init(&expected,world,env.horizon,env.fees,env.prior);
        unsigned int rng = 77;
        for (int step=0; step<125; step++) {
            float check[RES_OBS];res_observe(&expected,check);
            for (int i=0; i<RES_OBS; i++) assert(fabsf(check[i]-observations[i]) < 0.000001f);
            assert(env.agents[0].action_mask == NULL);
            actions[0] = (float)(rand_r(&rng)%RES_ACTIONS);
            float reward = res_step(&expected,(int)actions[0]);
            int done = expected.done;
            puf_step(&env);transitions++;
            assert(fabsf(rewards[0]-reward) < 0.000001f && terminals[0] == done);
            if (done) { episodes++;res_init(&expected,world,env.horizon,env.fees,env.prior); }
        }
        puf_close(&env);
        /* Dict is owned by the caller; release its tiny backing allocation. */
        free(kwargs.items);
    }
    printf("{\"transitions\":%d,\"episodes\":%d,\"status\":\"passed\",\"trainer_executed\":false}\n",transitions,episodes);
    return 0;
}
