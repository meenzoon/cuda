#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C"
__global__ void viterbi_kernel(
    int* observations, int* states, float* start_prob, float* trans_prob, float* emit_prob, int* path, float* delta, int* psi, int num_obs, int num_states) 
{
    int obs_idx = blockIdx.x;  // Observation index
    int state_idx = threadIdx.x;  // State index

    // Initialize delta and psi for the first observation
    if (obs_idx == 0) {
        delta[state_idx] = start_prob[state_idx] * emit_prob[state_idx * num_obs + observations[0]];
        psi[state_idx] = 0;
    } else {
        // For subsequent observations, calculate the maximum probability path
        float max_prob = -1.0f;
        int max_state = -1;

        for (int prev_state = 0; prev_state < num_states; ++prev_state) {
            float prob = delta[(obs_idx - 1) * num_states + prev_state] * trans_prob[prev_state * num_states + state_idx];
            if (prob > max_prob) {
                max_prob = prob;
                max_state = prev_state;
            }
        }

        delta[obs_idx * num_states + state_idx] = max_prob * emit_prob[state_idx * num_obs + observations[obs_idx]];
        psi[obs_idx * num_states + state_idx] = max_state;
    }

    // For the final step, find the most probable state
    if (obs_idx == num_obs - 1 && state_idx == 0) {
        float max_prob = -1.0f;
        int last_state = -1;

        for (int state = 0; state < num_states; ++state) {
            if (delta[(num_obs - 1) * num_states + state] > max_prob) {
                max_prob = delta[(num_obs - 1) * num_states + state];
                last_state = state;
            }
        }

        path[num_obs - 1] = last_state;

        // Backtrack to find the most probable path
        for (int t = num_obs - 2; t >= 0; --t) {
            path[t] = psi[(t + 1) * num_states + path[t + 1]];
        }
    }
}
