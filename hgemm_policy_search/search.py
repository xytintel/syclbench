import sys
import json
from subprocess import check_call, check_output
from configs import policies, print_policy


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
            print(string)
            good_policies.append(policy_id)
        else:
            break
    return good_policies


def main():
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    good_policies = run_and_select(m, n, k)
    policy_id = good_policies[0]
    print("starting point at m={}, n={}, k={}, policy:{}".format(m, n, k, print_policy(policy_id)))
    policy = policies[policy_id]

    begin_k = k
    while True:
        begin_k /= 2
        if int(begin_k) <= 0:
            break
        good_policies = run_and_select(m, n, begin_k)
        if policy_id not in good_policies:
            begin_k *= 2
            break
    print("begin_k = {}".format(begin_k))
    
    end_k = k
    while True:
        end_k *= 2
        if int(end_k) >= 32768:
            break
        good_policies = run_and_select(m, n, end_k)
        if policy_id not in good_policies:
            end_k /= 2
            break
    print("end_k = {}".format(end_k))

    begin_n = n
    while True:
        begin_n /= 2
        if int(begin_n) <= 0:
            break
        p0 = run_and_select(m, begin_n, begin_k)
        p1 = run_and_select(m, begin_n, end_k)
        if (policy_id not in p0) or (policy_id not in p1):
            begin_n *= 2
            break
    print("begin_n = {}".format(begin_n))

    end_n = n
    while True:
        end_n *= 2
        if int(end_n) >= 32768:
            break
        p0 = run_and_select(m, end_n, begin_k)
        p1 = run_and_select(m, end_n, end_k)
        if (policy_id not in p0) or (policy_id not in p1):
            begin_n /= 2
            break
    print("end_n = {}".format(end_n))

    begin_m = m
    while True:
        begin_m /= 2
        if int(begin_m) <= 0:
            break
        p0 = run_and_select(begin_m, begin_n, begin_k)
        p1 = run_and_select(begin_m, begin_n, end_k)
        p2 = run_and_select(begin_m, end_n, begin_k)
        p3 = run_and_select(begin_m, end_n, end_k)
        if (policy_id not in p0) or (policy_id not in p1) \
            or (policy_id not in p2) or (policy_id not in p3):
            begin_m *= 2
            break
    print("begin_m = {}".format(begin_m))

    end_m = m
    while True:
        end_m *= 2
        if int(end_m) >= 32768:
            break
        p0 = run_and_select(end_m, begin_n, begin_k)
        p1 = run_and_select(end_m, begin_n, end_k)
        p2 = run_and_select(end_m, end_n, begin_k)
        p3 = run_and_select(end_m, end_n, end_k)
        if (policy_id not in p0) or (policy_id not in p1) \
            or (policy_id not in p2) or (policy_id not in p3):
            end_m /= 2
            break
    print("end_m = {}".format(end_m))
    print("{}: {}<=m<={}, {}<=n<={}, {}<=k<={}".format(
        print_policy(policy_id), int(begin_m), int(end_m), int(begin_n), int(end_n), int(begin_k), int(end_k)))


if __name__ == '__main__':
    main()
