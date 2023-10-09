import json
from math import log
from functools import cmp_to_key
from subprocess import check_call, check_output
from configs import policies, mnk2policy


def policy_algo_baseline(m, n, k):
    policy = mnk2policy(m, n, k)
    if policy is not None:
        return policy
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


# def policy_algo_baseline(m, n, k):
#     def key_fn(key):
#         sgLimit = 32
#         xeCores = 64
#         (wgNegThreshold, reqNegThreshold, reqPosThreshold, sgNegThreshold, sgPerWgNegThresold) = (26, 160, 512, 192, 4)
#         (wgNegWeight, reqNegWeight, reqPosWeight, sgNegWeight, sgPerWgNegWeight, kSlicingWight, threadThrottlingWight, imbalanceWeight) = (1, 1, 0.5, 0.3, 0.1, 0.1, 0.05, 0.2)

#         wgM = key.wg_m
#         sgM = key.sg_m
#         wgN = key.wg_n
#         sgN = key.sg_n
#         sgK = key.sg_k
#         kSlicing = key.slm_ks

#         threadThrottling = 1
#         if (wgM / sgM) > int((m + sgM - 1) // sgM):
#             threadThrottling = (wgM / sgM) / int((m + sgM - 1) // sgM)
        
#         sgPerWg = (wgM / sgM) / threadThrottling * (wgN / sgN) * kSlicing
#         workgroups = int((m + (wgM - 1)) // wgM) * int((n + (wgN - 1)) // wgN)
#         subgroups = sgPerWg * workgroups
#         sgConcur = subgroups
#         imbalance = 0
#         if sgConcur > (sgLimit * xeCores / threadThrottling): 
#             if (sgConcur % (sgLimit * xeCores / threadThrottling)) != 0:
#                 imbalance = 1
#             sgConcur = sgLimit * xeCores / threadThrottling
#         reqs = sgConcur * sgK / 32 * sgN / 32

#         score = 0
#         if workgroups < wgNegThreshold:
#             score -= ((log(wgNegThreshold / workgroups) / log(2)) * wgNegWeight)
#         if reqs < reqNegThreshold:
#             score -= ((log(reqNegThreshold / reqs) / log(2)) * reqNegWeight)
#         if reqs > reqPosThreshold:
#             score -= ((log(reqs / reqPosThreshold) / log(2)) * reqPosWeight)
#         if subgroups < sgNegThreshold:
#             score -= ((log(sgNegThreshold / subgroups) / log(2)) * sgNegWeight)
#         if sgPerWg < sgPerWgNegThresold:
#             score -= ((log(sgPerWgNegThresold / sgPerWg) / log(2)) * sgPerWgNegThresold)
#         score -= ((log(kSlicing) / log(2)) * kSlicingWight)
#         score += ((log(threadThrottling) / log(2)) * threadThrottlingWight)
#         score -= (imbalance * imbalanceWeight)

#         return -score

#     sorted_policies = sorted(policies, key=key_fn)
#     return sorted_policies[0]


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
    check_call(['bash', 'build.sh', './hgemm_qkv_policy_search/hgemm_qkv_xetla.cpp'])
    shapes = get_all_shapes('./hgemm_qkv_policy_search/focus_shapes.txt')
    # shapes = shapes[:20]
    hit = 0
    num_samples = 0
    with open('not_hit.log', 'w') as f:
        for shape in shapes:
            try:
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
                timems_pred = 0
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
                        if pred_policy.id == policy_id:
                            timems_pred = timems
                string = "pred_{}_hgemm_policy::_{}x{}_{}x{}x{}_{}_true_:{}".format(pred_policy.id, pred_policy.wg_m, 
                            pred_policy.wg_n, pred_policy.sg_m, pred_policy.sg_n, pred_policy.sg_k, 
                            pred_policy.slm_ks, timems_pred)
                print(string)
                num_samples += 1
                if pred_policy.id in good_policies:
                    hit += 1
                else:
                    res = policies[good_policies[0]]
                    key = "{{{}, {}, {}}}".format(m, n, k)
                    string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(
                        res.wg_m, res.wg_n, res.sg_m, res.sg_n, res.sg_k, res.slm_ks)
                    info = "{{{}, {}}},\n".format(key, string)
                    f.write(info)
                    f.flush()
            except Exception as e:
                pass

    hit_rate = hit / num_samples
    print("hit_rate:{}".format(hit_rate))


if __name__ == '__main__':
    main()
