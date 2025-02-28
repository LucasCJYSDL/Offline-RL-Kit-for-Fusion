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
        void prepare(const vector[vector[float]] &priors, const vector[vector[float]] &states, const vector[vector[float]] &pi_mus, const vector[vector[float]] &pi_stds)
        vector[vector[int]] get_distributions()
        vector[vector[vector[float]]] get_sampled_actions()
        vector[float] get_target_values()
        vector[vector[float]] get_target_qs()
    
    cdef cppclass CSearchResults:
        CSearchResults()
        CSearchResults(int num)
        vector[vector[float]] last_states, last_priors, last_actions
        vector[bool] dones
    
    void cbatch_traverse(CRoots *roots, float alpha, float beta, float gamma, float lambd, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)

    void cbatch_backpropagate(const vector[vector[float]] &priors, const vector[vector[float]] &states, float discount_factor, const vector[float] &rewards, const vector[float] &values, const vector[bool] &dones, const vector[vector[float]] &mus, const vector[vector[float]] &stds, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);


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
    
    def prepare(self, list priors, list states, list policy_logits_pool):
        self.roots[0].prepare(priors, states, policy_logits_pool[0], policy_logits_pool[1])

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

    return results.cresults.last_priors, results.cresults.last_states, results.cresults.last_actions, results.cresults.dones


def batch_backpropagate(list priors, list states, float discount_factor, list rewards, list values, list dones, list mus, list stds,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results):

    cdef vector[vector[float]] cpriors = priors
    cdef vector[vector[float]] cstates = states
    cdef vector[float] crewards = rewards
    cdef vector[float] cvalues = values
    cdef vector[bool] cdones = dones
    cdef vector[vector[float]] cmus = mus
    cdef vector[vector[float]] cstds = stds

    cbatch_backpropagate(cpriors, cstates, discount_factor, crewards, cvalues, cdones, cmus, cstds,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults)