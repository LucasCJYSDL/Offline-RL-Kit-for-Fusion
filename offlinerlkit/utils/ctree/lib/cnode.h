#ifndef CNODE_H
#define CNODE_H

#include <vector>
#include <stdlib.h>
#include <map>

#include "cminimax.h"


class CNode
{
public:
    int state_num;
    std::vector<std::vector<float> > prior_list;
    std::vector<std::vector<float> > state_list;
    std::vector<int> state_visit_counts;
    std::vector<int> action_nums;
    std::vector<bool> done_list;

    std::vector<float> reward_list; // r(s'', a'', s)
    std::vector<float> value_sum_list; // V(s)
    std::vector<std::vector<float> > q_value_sum_list; //Q(s, a)

    std::vector<std::vector<std::vector<float> > > legal_actions_list;
    std::vector<std::vector<CNode *> > children_list;
    std::vector<std::vector<int> > state_action_visit_counts;

    int state_idx;
    int action_idx;

    int num_of_sampled_actions;
    int num_of_sampled_states;

    // std::vector<float> state_mu;
    // std::vector<float> state_std;
    std::vector<std::vector<float> > action_mus;
    std::vector<std::vector<float> > action_stds;

    CNode();
    // sampled related core code
    CNode(int num_of_sampled_actions, int num_of_sampled_states);
    ~CNode();

    void expand(const std::vector<float> &prior, const std::vector<float> &state, const std::vector<float> &pi_mus, const std::vector<float> &pi_log_stds, float reward, bool is_done);
    bool expanded();
    void sample_action();

    float q_value();
    void select_action(CMinMaxStats &min_max_stats, float lambd);
    float ucb_score(int child_idx, CMinMaxStats &min_max_stats, float lambd);
    // sampled related core code
    // std::vector<std::vector<float> > get_trajectory();
    // std::vector<int> get_children_distribution();
    // CNode *get_child(CAction action);
};

class CRoots
{
public:
    int root_num;
    std::vector<CNode> roots;

    CRoots();
    CRoots(int root_num, int num_of_sampled_actions, int num_of_sampled_states);
    ~CRoots();
    void prepare(const std::vector<std::vector<float> > &priors, const std::vector<std::vector<float> > &states, const std::vector<std::vector<float> > &pi_mus, const std::vector<std::vector<float> > &pi_stds);
    void clear();

    // sampled related core code
    // std::vector<std::vector<std::vector<float> > > get_trajectories();
    std::vector<std::vector<std::vector<float> > > get_sampled_actions();
    std::vector<std::vector<int> > get_distributions();
    std::vector<float> get_target_values();
    std::vector<std::vector<float> > get_target_qs();
};

class CSearchResults
{
public:
    int num;
    std::vector<std::vector<float> > last_priors;
    std::vector<std::vector<float> > last_states;
    std::vector<std::vector<float> > last_actions;
    std::vector<bool> dones;

    std::vector<CNode *> nodes;
    std::vector<std::vector<CNode *> > search_paths;

    CSearchResults();
    CSearchResults(int num);
    ~CSearchResults();
};

void cbatch_traverse(CRoots *roots, float alpha, float beta, float gamma, float lambd, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);

void cbackpropagate(std::vector<CNode *> &search_path, CMinMaxStats &min_max_stats, float reward, float value, float discount_factor);

void cbackpropagate_when_done(std::vector<CNode *> &search_path, CMinMaxStats &min_max_stats, float discount_factor);

void cbatch_backpropagate(const std::vector<std::vector<float> > &priors, const std::vector<std::vector<float> > &states, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<bool> &dones, const std::vector<std::vector<float> > &mus, const std::vector<std::vector<float> > &stds, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);

#endif