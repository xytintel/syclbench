import sys
import json
from subprocess import check_call, check_output
from configs import policies, print_policy
from evaluate import get_all_shapes


def sort_out_policy(res):
    jsonlines = []
    for line in res.split('\n'):
        if line.startswith('{'):
            js = json.loads(line.strip())
            jsonlines.append(js)
    jsonlines = sorted(jsonlines, key=lambda x: x['timems'])
    return jsonlines


def run_and_select(m, n, k):
    output = check_output(['./a.out', str(m), str(n), str(k)])
    output = output.decode('utf-8')
    real_policy_rank = sort_out_policy(output)
    timems_best = real_policy_rank[0]['timems']
    good_policies = []
    for i, item in enumerate(real_policy_rank):
        timems = item['timems']
        policy_id = item['policy']
        diff = abs(timems - timems_best) / timems_best
        if diff < 0.05:
            string = "--- m={},n={},k={} real_{}_hgemm_policy::_{}x{}_{}x{}x{}_{}_true_:{}".format(
                m, n, k, policy_id, item['WG_M'], 
                item['WG_N'], item['SG_M'], item['SG_N'], item['SG_K'], item['SLM_KS'],
                item['timems'])
            # print(string)
            good_policies.append(policy_id)
        else:
            break
    return good_policies


def main():
    check_call(['bash', 'build.sh', './hgemm_policy_search/hgemm_xetla.cpp'])
    id_base = 0
    ms = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    ns = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
    ks = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768]
    shapes = []
    for m in ms:
        for n in ns:
            for k in ks:
                shapes.append([m, n, k])

    with open('hgemm_policy_search.log', 'w') as f:
        for i, shape in enumerate(shapes):
            m = shape[0]
            n = shape[1]
            k = shape[2]
            try:
                res = policies[run_and_select(m, n, k)[0]]
                key = "{{{}, {}, {}}}".format(m, n, k)
                string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(
                    res.wg_m, res.wg_n, res.sg_m, res.sg_n, res.sg_k, res.slm_ks)
                item = "{{{}, {}}}, // {}\n".format(key, string, i)
                # print(item)
                f.write(item)
                f.flush()
            except Exception as e:
                pass


if __name__ == '__main__':
    main()
