import json
from functools import cmp_to_key
from subprocess import check_call, check_output
from configs import policies, mnk2_policy_id
import bisect


def policy_algo_baseline(m, n, k):

    if m <= 128 and n <= 32768 and k <=32768:
        ms = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        ns = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
        ks = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
        m_ = ms[bisect.bisect_right(ms, m) - 1]
        n_ = ns[bisect.bisect_right(ns, n) - 1]
        k_ = ks[bisect.bisect_right(ks, k) - 1]
        key = "{}, {}, {}".format(m_, n_, k_)
        policy_id = mnk2_policy_id.get(key, None)
        if policy_id is not None:
            return policies[policy_id]

    print('--- using algo')
    def compare_fn(lhs, rhs):
        TOTAL_SS = 64
        lhs_num_ss, lhs_wg_eff, lhs_aspect_r = lhs.traits(m, n, k)
        rhs_num_ss, rhs_wg_eff, rhs_aspect_r = rhs.traits(m, n, k)
        lss = abs(lhs_num_ss - TOTAL_SS);
        rss = abs(rhs_num_ss - TOTAL_SS);
        if lss != rss:
            return -1 if lss < rss else 1
        elif lhs_wg_eff != rhs_wg_eff:
            return -1 if lhs_wg_eff > rhs_wg_eff else 1
        else:
            return -1 if lhs_aspect_r < rhs_aspect_r else 1
    sorted_policies = sorted(policies, key=cmp_to_key(compare_fn))
    return sorted_policies[0]


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


def main():
    check_call(['bash', 'build.sh', './hgemm_policy_search/hgemm_xetla.cpp'])
    shapes = get_all_shapes('./hgemm_policy_search/focus_shapes.txt')
    # shapes = shapes[:20]
    hit = 0
    for shape in shapes:
        m = shape[0]
        n = shape[1]
        k = shape[2]
        print("collecting: m={}, n={}, k={}".format(m, n, k))
        pred_policy = policy_algo_baseline(m, n, k)
        output = check_output(['./a.out', str(m), str(n), str(k)])
        output = output.decode('utf-8')
        real_policy_rank = sort_out_policy(output)
        good_policies = []
        timems_best = real_policy_rank[0]['timems']
        for i, item in enumerate(real_policy_rank):
            timems = item['timems']
            policy_id = item['policy']
            diff = abs(timems - timems_best) / timems_best
            if diff < 0.1:
                string = "real_{}_hgemm_policy::_{}x{}_{}x{}x{}_{}_true_:{}".format(policy_id, item['WG_M'], 
                    item['WG_N'], item['SG_M'], item['SG_N'], item['SG_K'], item['SLM_KS'],
                    item['timems'])
                print(string)
                good_policies.append(policy_id)
            else:
                break
        string = "pred_{}_hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(pred_policy.id, pred_policy.wg_m, 
                    pred_policy.wg_n, pred_policy.sg_m, pred_policy.sg_n, pred_policy.sg_k, 
                    pred_policy.slm_ks)
        print(string)
        if pred_policy.id in good_policies:
            hit += 1
    hit_rate = hit / len(shapes)
    print("hit_rate:{}".format(hit_rate))


if __name__ == '__main__':
    main()
