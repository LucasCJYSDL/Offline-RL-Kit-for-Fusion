# distutils:language=c++
# cython:language_level=3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "./lib/cminimax.h":
    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList(int num)
        void set_delta(float value_delta_max)

cdef extern from "./lib/cnode.h":
    cdef cppclass CRoots:
        CRoots(int root_num, int num_of_sampled_actions, int num_of_sampled_states)
        void prepare(const vector[vector[float]] &priors, const vector[vector[float]] &states, const vector[vector[float]] &pi_mus, const vector[vector[float]] &pi_stds, const vector[vector[float]] &full_states, const vector[vector[float]] &pre_actions, const vector[int] &time_steps)
        vector[vector[int]] get_distributions()
        vector[vector[vector[float]]] get_sampled_actions()
        vector[float] get_target_values()
        vector[vector[float]] get_target_qs()
    
    cdef cppclass CSearchResults:
        CSearchResults()
        CSearchResults(int num)
        vector[vector[float]] last_states, last_priors, last_actions, last_full_states, last_pre_actions
        vector[int] last_time_steps
        vector[bool] dones
    
    void cbatch_traverse(CRoots *roots, float alpha, float beta, float gamma, float lambd, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)

    void cbatch_backpropagate(const vector[vector[float]] &priors, const vector[vector[float]] &states, const vector[vector[float]] &full_states, const vector[vector[float]] &pre_actions, const vector[int] &time_steps, float discount_factor, const vector[float] &rewards, const vector[float] &values, const vector[bool] &dones, const vector[vector[float]] &mus, const vector[vector[float]] &stds, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst

cdef class Roots:
    cdef int root_num
    cdef CRoots *roots

    def __cinit__(self):
        pass

    def __cinit__(self, int root_num, int num_of_sampled_actions, int num_of_sampled_states):
        self.root_num = root_num
        self.roots = new CRoots(root_num, num_of_sampled_actions, num_of_sampled_states)
    
    def prepare(self, list priors, list states, list policy_logits_pool, list full_states, list pre_actions, list time_steps):
        self.roots[0].prepare(priors, states, policy_logits_pool[0], policy_logits_pool[1], full_states, pre_actions, time_steps)

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_sampled_actions(self):
        return self.roots[0].get_sampled_actions()
    
    def get_target_values(self):
        return self.roots[0].get_target_values()

    def get_target_qs(self):
        return self.roots[0].get_target_qs()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


def batch_traverse(Roots roots, float alpha, float beta, float gamma, float lambd, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results):
    cbatch_traverse(roots.roots, alpha, beta, gamma, lambd, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

    return results.cresults.last_priors, results.cresults.last_states, results.cresults.last_actions, \
           results.cresults.last_full_states, results.cresults.last_pre_actions, results.cresults.last_time_steps, results.cresults.dones


def batch_backpropagate(list priors, list states, list full_states, list pre_actions, list time_steps, 
                        float discount_factor, list rewards, list values, list dones, list mus, list stds,
                        MinMaxStatsList min_max_stats_lst, ResultsWrapper results):

    cdef vector[vector[float]] cpriors = priors
    cdef vector[vector[float]] cstates = states

    cdef vector[vector[float]] cfull_states = full_states
    cdef vector[vector[float]] cpre_actions = pre_actions
    cdef vector[int] ctime_steps = time_steps

    cdef vector[float] crewards = rewards
    cdef vector[float] cvalues = values
    cdef vector[bool] cdones = dones
    cdef vector[vector[float]] cmus = mus
    cdef vector[vector[float]] cstds = stds

    cbatch_backpropagate(cpriors, cstates, cfull_states, cpre_actions, ctime_steps, 
                         discount_factor, crewards, cvalues, cdones, cmus, cstds,
                         min_max_stats_lst.cmin_max_stats_lst, results.cresults)