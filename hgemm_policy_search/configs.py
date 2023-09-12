import os


config_count = 0


class GemmConfig:
    def __init__(self, wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks):
        self.wg_m = wg_m
        self.wg_n = wg_n
        self.sg_m = sg_m
        self.sg_n = sg_n
        self.sg_k = sg_k
        self.slm_ks = slm_ks
        global config_count
        self.id = config_count
        config_count += 1

    def traits(self, m, n, k):
        ms = (m + self.wg_m - 1) // self.wg_m
        ns = (n + self.wg_n - 1) // self.wg_n
        num_ss = ms * ns
        if m > self.wg_m:
            vm = self.wg_m
        else:
            vm = m
        if n > self.wg_n:
            vn = self. wg_n
        else:
            vn = n
        wg_eff = vm * vn / self.wg_m / self.wg_n
        aspect_r = max(self.wg_m / self.wg_n, self.wg_n / self.wg_m)
        r = {
            'num_ss': num_ss,
            'wg_eff': wg_eff,
            'aspect_r': aspect_r,
        }
        return r


policies = []
with open('hgemm_policy_search/hgemm_xetla.cpp', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('  _('):
            args = []
            for arg in line[line.find('(')+1:line.find(')')].strip().split(','):
                args.append(int(arg.strip()))
            policies.append(GemmConfig(args[0], args[1], args[2], args[3], args[4], args[5]))
print('total policies:', len(policies))


def print_policy(idx):
    policy = policies[idx]
    string = "<{}>hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(idx, policy.wg_m, 
                    policy.wg_n, policy.sg_m, policy.sg_n, policy.sg_k, policy.slm_ks)
    return string
