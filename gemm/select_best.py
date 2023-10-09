import sys
import json
from tqdm import tqdm
from subprocess import check_output


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
        return num_ss, wg_eff, aspect_r


policies = []
with open('gemm/gemm_config.h', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('  _('):
            args = []
            for arg in line[line.find('(')+1:line.find(')')].strip().split(','):
                args.append(int(arg.strip()))
            policies.append(GemmConfig(args[0], args[1], args[2], args[3], args[4], args[5]))
print('total policies:', len(policies))


def get_all_shapes(file):
    shapes = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 2 and not line.startswith('# '):
                line_ = line.split(',')
                m = line_[0].strip()
                n = line_[1].strip()
                k = line_[2].strip()
                k = k.split(' ')[0].strip()
                shapes.append([int(m), int(n), int(k)])
    return shapes


def sort_out_policy(res):
    jsonlines = []
    for line in res.split('\n'):
        if line.startswith('{'):
            js = json.loads(line.strip())
            jsonlines.append(js)
    jsonlines = sorted(jsonlines, key=lambda x: x['timems'])
    return jsonlines


def run_and_select(m, n, k, f):
    output = check_output(['./a.out', str(m), str(n), str(k)])
    output = output.decode('utf-8')
    f.write(output)
    f.flush()
    real_policy_rank = sort_out_policy(output)
    # timems_best = real_policy_rank[0]['timems']
    policy_id_best = real_policy_rank[0]['policy']
    return policy_id_best


def main():
    shapes = get_all_shapes('./gemm/focus_shapes.txt')
    pbar = tqdm(total=len(shapes), ncols=80)
    with open('policy_selection_raw.log', 'w') as rf:
        with open('policy_selection.log', 'w') as f:
            for i, shape in enumerate(shapes):
                pbar.update(1)
                m = shape[0]
                n = shape[1]
                k = shape[2]
                try:
                    res = policies[run_and_select(m, n, k, rf)]
                    key = "{{{}, {}, {}}}".format(m, n, k)
                    string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(
                        res.wg_m, res.wg_n, res.sg_m, res.sg_n, res.sg_k, res.slm_ks)
                    item = "{{{}, {}}}, // {}\n".format(key, string, i)
                    f.write(item)
                    f.flush()
                except Exception as e:
                    print("error: m={}, n={}, k={}".format(m, n, k))
                    print(e)
                    pass
    pbar.close()


if __name__ == '__main__':
    main()
