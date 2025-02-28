#include <stdlib.h>
#include "cnode.h"
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <cmath>
#include <cassert>


CNode::CNode()
{
    /*
    Overview:
        Initialization of CNode.
    */
    this->num_of_sampled_actions = 20;
    this->num_of_sampled_states = 5;
    this->state_idx = -1;
    this->action_idx = -1;
    this->state_num = 0;
}

CNode::CNode(int num_of_sampled_actions, int num_of_sampled_states)
{
    /*
    Overview:
        Initialization of CNode with prior, legal actions, action_space_size, num_of_sampled_actions, continuous_action_space.
    Arguments:
        - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
        - ...
    */
    this->num_of_sampled_actions = num_of_sampled_actions;
    this->num_of_sampled_states = num_of_sampled_states;
    this->state_idx = -1;
    this->action_idx = -1;
    this->state_num = 0;
}

void CNode::expand(const std::vector<float> &prior, const std::vector<float> &state, const std::vector<float> &pi_mus, const std::vector<float> &pi_stds, float reward, bool is_done)
{
    /*
    Overview:
        Expand the child nodes of the current node.
    Arguments:
        - ...
    */
    // TODO: reserve

    this->prior_list.push_back(prior);
    this->state_list.push_back(state);
    this->state_num += 1;
    this->state_visit_counts.push_back(0);
    this->action_nums.push_back(0);
    this->done_list.push_back(is_done);

    this->reward_list.push_back(reward); // for non-root nodes, this value would be changed in cbackpropagate
    this->value_sum_list.push_back(0);
    this->q_value_sum_list.push_back(std::vector<float>());

    this->legal_actions_list.push_back(std::vector<std::vector<float> >());
    this->children_list.push_back(std::vector<CNode *>());
    this->state_action_visit_counts.push_back(std::vector<int>());

    this->action_mus.push_back(pi_mus);
    this->action_stds.push_back(pi_stds);
}

bool CNode::expanded()
{
    /*
    Overview:
        Return whether the current node is expanded.
    */
    return not this->state_list.empty();
}

void CNode::sample_action()
{   
    std::vector<float> pi_mus = this->action_mus[this->state_idx];
    std::vector<float> pi_stds = this->action_stds[this->state_idx];
    int action_num = pi_mus.size();
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    float sampled_action_one_dim_before_tanh;
    // warning: not every env should apply tanh to the action output
    std::vector<float> sampled_action_after_tanh;
    for (int j = 0; j < action_num; ++j)
    {   
        std::normal_distribution<float> distribution(pi_mus[j], pi_stds[j]);
        sampled_action_one_dim_before_tanh = distribution(generator);
        sampled_action_after_tanh.push_back(tanh(sampled_action_one_dim_before_tanh));
    }
    this->legal_actions_list[this->state_idx].push_back(sampled_action_after_tanh);
    this->children_list[this->state_idx].push_back(new CNode(this->num_of_sampled_actions, this->num_of_sampled_states));
    this->state_action_visit_counts[this->state_idx].push_back(0);
    this->q_value_sum_list[this->state_idx].push_back(0.0);
    this->action_idx = this->action_nums[this->state_idx];
    this->action_nums[this->state_idx] += 1;
}

float CNode::q_value()
{
    /*
    Overview:
        Return the real value of the current tree.
    */
    if (this->state_action_visit_counts[this->state_idx][this->action_idx] == 0)
    {
        return 0.0;
    }
    return float(this->q_value_sum_list[this->state_idx][this->action_idx]) / float(this->state_action_visit_counts[this->state_idx][this->action_idx]);
}

void CNode::select_action(CMinMaxStats &min_max_stats, float lambd)
{
    /*
        Overview:
            Select the child node of the roots according to ucb scores.
        Arguments:
            - min_max_stats: a tool used to min-max normalize the score.
            - ...
    */
    int children_size = this->action_nums[this->state_idx];
    assert(children_size > 1 && "The size of the children space has to be larger than 1 when entering this function.");

    float max_score = FLOAT_MIN;
    const float epsilon = 0.000001;
    std::vector<int> max_index_lst;

    // TODO: simply using argmax, which can improve the speed.
    for (int i=0; i < children_size; ++i)
    {
        float temp_score = this->ucb_score(i, min_max_stats, lambd);

        if (max_score < temp_score)
        {
            max_score = temp_score;
            max_index_lst.clear();
            max_index_lst.push_back(i);
        }
        else if (temp_score >= max_score - epsilon)
        {
            max_index_lst.push_back(i);
        }
    }

    int candidate_size = max_index_lst.size();
    assert(candidate_size > 0 && "There should be at least one candidates.");
    int rand_index = rand() % candidate_size;
    this->action_idx = max_index_lst[rand_index];
}

float CNode::ucb_score(int child_idx, CMinMaxStats &min_max_stats, float lambd)
{
    int n_s = this->state_visit_counts[this->state_idx];
    int n_sa = this->state_action_visit_counts[this->state_idx][child_idx];
    assert(n_sa > 0);
    float exploration_term = lambd * sqrt(log(float(n_s))/float(n_sa));

    float exploiation_term = this->q_value_sum_list[this->state_idx][child_idx] / float(n_sa);
    exploiation_term = min_max_stats.normalize(exploiation_term);
    // TODO
    // if (exploiation_term < 0)
    //     exploiation_term = 0;
    // if (exploiation_term > 1)
    //     exploiation_term = 1;
    // std::cout<< exploiation_term << " " << exploration_term << std::endl;
    return exploiation_term + exploration_term;
}

CNode::~CNode() {
    // Iterate through the outer vector
    for (auto& inner_vector : children_list) {
        // Iterate through the inner vector
        for (CNode* node : inner_vector) {
            delete node;  // Free the memory for each CNode*
        }
        inner_vector.clear();  // Clear the inner vector after deleting the nodes
    }
    children_list.clear();  // Clear the outer vector to clean up any remaining references
}

CRoots::CRoots()
{
    this->root_num = 0;
}

CRoots::CRoots(int root_num, int num_of_sampled_actions, int num_of_sampled_states)
{
    /*
    Overview:
        Initialization of CNode with root_num, legal_actions_list, action_space_size, num_of_sampled_actions, continuous_action_space.
    Arguments:
        - root_num: the number of the current root.
        - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
        - ...
    */
    this->root_num = root_num;

    for (int i = 0; i < this->root_num; ++i)
    {
        this->roots.push_back(CNode(num_of_sampled_actions, num_of_sampled_states));
    }
}

void CRoots::clear()
{
    this->roots.clear();
}

void CRoots::prepare(const std::vector<std::vector<float> > &priors, const std::vector<std::vector<float> > &states, const std::vector<std::vector<float> > &pi_mus, const std::vector<std::vector<float> > &pi_stds)
{
    /*
    Overview:
        Expand the roots.
    Arguments:
        - policies: the vector of policy logits of each root, including mus and log_stds.
        - ...
    */
    // sampled related core code
    for (int i = 0; i < this->root_num; ++i)
    {   
        this->roots[i].expand(priors[i], states[i], pi_mus[i], pi_stds[i], 0.0, false); // by default, the root node is not done
    }
}

std::vector<std::vector<int> > CRoots::get_distributions()
{
    /*
    Overview:
        Get the children distribution of each root.
    Returns:
        - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
    */
    std::vector<std::vector<int> > distributions;
    distributions.reserve(this->root_num);

    for (int i = 0; i < this->root_num; ++i)
    {
        distributions.push_back(this->roots[i].state_action_visit_counts[0]);
    }
    return distributions;
}

std::vector<std::vector<std::vector<float> > > CRoots::get_sampled_actions()
{
    /*
    Overview:
        Get the sampled_actions of each root.
    Returns:
        - python_sampled_actions: a vector of sampled_actions for each root, e.g. the size of original action space is 6, the K=3, 
        python_sampled_actions = [[1,3,0], [2,4,0], [5,4,1]].
    */
    std::vector<std::vector<std::vector<float> > > python_sampled_actions;
    // TODO: use reserve
    for (int i = 0; i < this->root_num; ++i)
    {
        python_sampled_actions.push_back(this->roots[i].legal_actions_list[0]);
    }

    return python_sampled_actions;
}

std::vector<float> CRoots::get_target_values()
{
    /*
        Overview: Return the estimated value of each root.
    */
    std::vector<float> values;
    values.reserve(this->root_num);
    for (int i = 0; i < this->root_num; ++i)
    {
        values.push_back(this->roots[i].value_sum_list[0] / float(this->roots[i].state_visit_counts[0]));
    }
    return values;
}

std::vector<std::vector<float> > CRoots::get_target_qs()
{
    /*
        Overview: Return the estimated qs at each root.
    */
   std::vector<std::vector<float> > qs;
    for (int i = 0; i < this->root_num; ++i)
    {
        qs.push_back(std::vector<float>());
        int a_len = this->roots[i].q_value_sum_list[0].size();
        for (int j = 0; j < a_len; ++j)
        {
            qs[i].push_back(this->roots[i].q_value_sum_list[0][j] / float(this->roots[i].state_action_visit_counts[0][j]));
        }
    }
    return qs;
}

CRoots::~CRoots() {}

CSearchResults::CSearchResults()
{
    /*
    Overview:
        Initialization of CSearchResults, the default result number is set to 0.
    */
    this->num = 0;
}

CSearchResults::CSearchResults(int num)
{
    /*
    Overview:
        Initialization of CSearchResults with result number.
    */
    this->num = num;
    this->search_paths.reserve(num+1);
    for (int i = 0; i < num; ++i)
    {
        this->search_paths.push_back(std::vector<CNode *>());
    }
}

CSearchResults::~CSearchResults() {}

void get_time_and_set_rand_seed()
{
#ifdef _WIN32
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  ULARGE_INTEGER uli;
  uli.LowPart = ft.dwLowDateTime;
  uli.HighPart = ft.dwHighDateTime;
  uint64_t timestamp = (uli.QuadPart - 116444736000000000ULL) / 10000000ULL;
  srand(timestamp % RAND_MAX);
#else
    timeval tv;
    gettimeofday(&tv, nullptr);
    srand(tv.tv_usec);
#endif
}

void cbatch_traverse(CRoots *roots, float alpha, float beta, float gamma, float lambd, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
{
    /*
    Overview:
        Search node path from the roots.
    Arguments:
        - roots: the roots that search from.
        - alpha, beta, gamma, lambda: hyperparameters for the rollout process.
        - disount_factor: the discount factor of reward.
        - min_max_stats: a tool used to min-max normalize the score.
        - results: the search results.
    */
    // set seed
    get_time_and_set_rand_seed();
    std::vector<float> last_prior, last_state, last_action;

    for (int i = 0; i < results.num; ++i)
    {
        CNode *node = &(roots->roots[i]);
        node->state_idx = 0;
        results.search_paths[i].push_back(node);
        bool is_children_state_sufficient = true;
        bool is_done = node->done_list[node->state_idx];

        while (node->expanded() && is_children_state_sufficient && (not is_done))
        {
            // should only calculate this if action_sofar < node->num_of_sampled_actions
            int visit_sofar = std::floor(std::pow(node->state_visit_counts[node->state_idx], alpha));
            int action_sofar = node->action_nums[node->state_idx];
            // get node->action_idx
            if (visit_sofar >= action_sofar && action_sofar < node->num_of_sampled_actions)
            {
                node->sample_action();
            } else{
                // sample from existing actions
                node->select_action(min_max_stats_lst->stats_lst[i], lambd);
            }
            // std::cout << node->state_idx << " " << node->action_idx << " " << visit_sofar << " " << action_sofar << std::endl;
            last_prior = node->prior_list[node->state_idx];
            last_state = node->state_list[node->state_idx];
            last_action = node->legal_actions_list[node->state_idx][node->action_idx];
            int action_visit_sofar = std::floor(std::pow(node->state_action_visit_counts[node->state_idx][node->action_idx], beta));

            node = node->children_list[node->state_idx][node->action_idx];
            if  (node->expanded())
            {
                // sample the next state and assign node->state_idx
                int children_state_sofar = node->state_num;
                if (action_visit_sofar >= children_state_sofar && children_state_sofar < node->num_of_sampled_states)
                {
                    // std::cout << "sample_new " << action_visit_sofar << " " << children_state_sofar << std::endl;
                    is_children_state_sufficient = false;
                } else {
                    int argmin = std::distance(node->state_visit_counts.begin(), std::min_element(node->state_visit_counts.begin(), node->state_visit_counts.end()));
                    node->state_idx = argmin;
                    // std::cout << "select from exisiting ones " << argmin << std::endl;
                    is_done = node->done_list[argmin];
                }
            } 
            results.search_paths[i].push_back(node);
        }
        results.last_priors.push_back(last_prior);
        results.last_states.push_back(last_state);
        results.last_actions.push_back(last_action);
        results.nodes.push_back(node);
        results.dones.push_back(is_done);
        // std::cout << "end of " << i << std::endl;
    }
}

void cbatch_backpropagate(const std::vector<std::vector<float> > &priors, const std::vector<std::vector<float> > &states, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<bool> &dones, const std::vector<std::vector<float> > &mus, const std::vector<std::vector<float> > &stds, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
{
    /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - discount_factor: the discount factor of reward.
            - values: the values to propagate along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - ...
    */
    int j = 0;
    for (int i = 0; i < results.num; ++i)
    {
        if (not results.dones[i])
        {
            results.nodes[i]->expand(priors[j], states[j], mus[j], stds[j], rewards[j], dones[j]);
            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], rewards[j], values[j], discount_factor);
            j += 1;
        } else {
            cbackpropagate_when_done(results.search_paths[i], min_max_stats_lst->stats_lst[i], discount_factor);
        }
        // std::cout<< "end of " << i << std::endl;
    }
}

void cbackpropagate(std::vector<CNode *> &search_path, CMinMaxStats &min_max_stats, float reward, float value, float discount_factor)
{
    /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
            - ...
    */

    float bootstrap_value = value;
    float next_reward = reward;
    int start_idx = search_path.size() - 2;

    for (int i = start_idx; i >= 0; --i)
    {
        CNode *node = search_path[i];
        node->state_visit_counts[node->state_idx] += 1;
        node->state_action_visit_counts[node->state_idx][node->action_idx] += 1;
        // std::cout << node->state_idx << " " << node->action_idx << std::endl;
        bootstrap_value = next_reward + discount_factor * bootstrap_value;
        node->value_sum_list[node->state_idx] += bootstrap_value;
        node->q_value_sum_list[node->state_idx][node->action_idx] += bootstrap_value;
        next_reward = node->reward_list[node->state_idx];

        min_max_stats.update(node->q_value());
        // TODO: min_max_stats.update(next_reward + discount_factor * node->value());

        node->state_idx = -1;
        node->action_idx = -1;
    }
}

void cbackpropagate_when_done(std::vector<CNode *> &search_path, CMinMaxStats &min_max_stats, float discount_factor)
{
    // deal with the last node
    int path_end_idx = search_path.size() - 1;
    CNode *last_node = search_path[path_end_idx];
    assert(last_node->done_list[last_node->state_idx] && last_node->action_idx == -1);

    last_node->state_visit_counts[last_node->state_idx] += 1; 
    float bootstrap_value = 0.0; // its value sum is always 0, so doesn't need update
    float next_reward = last_node->reward_list[last_node->state_idx];
    last_node->state_idx = -1;

    for (int i = path_end_idx-1; i >= 0; --i)
    {
        CNode *node = search_path[i];
        node->state_visit_counts[node->state_idx] += 1;
        node->state_action_visit_counts[node->state_idx][node->action_idx] += 1;

        bootstrap_value = next_reward + discount_factor * bootstrap_value;
        node->value_sum_list[node->state_idx] += bootstrap_value;
        node->q_value_sum_list[node->state_idx][node->action_idx] += bootstrap_value;
        next_reward = node->reward_list[node->state_idx];
       
        min_max_stats.update(node->q_value());

        node->state_idx = -1;
        node->action_idx = -1;
    }
}
