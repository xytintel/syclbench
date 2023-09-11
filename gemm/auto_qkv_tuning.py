import json
from subprocess import check_call, check_output


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
                shapes.append([m, n, k])
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
    shapes = get_all_shapes('./gemm/focus_shapes.txt')
    check_call(['bash', 'build.sh', './gemm/hgemm_qkv_xetla.cpp'])
    datas = []
    for shape in shapes:
        m = shape[0]
        n = shape[1]
        k = shape[2]
        print("collecting: m={}, n={}, k={}".format(m, n, k))
        try:
            output = check_output(['./a.out', str(m), str(n), str(k)])
            output = output.decode('utf-8')
            datas.append(sort_out_policy(output))
        except Exception as e:
            pass
    policy_count = {}
    mnk2policy = {}
    for data in datas:
        # {"m=1, n=7168, k=14336", hgemm_policy::_8x256_8x16x16_2_true_},
        key = "{{{}, {}, {}}}".format(data[0]['m'], data[0]['n'], data[0]['k'])
        cur = data[0]
        string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(cur['WG_M'], cur['WG_N'], cur['SG_M'], cur['SG_N'], cur['SG_K'], cur['SLM_KS'])
        item = "{{{}, {}}},".format(key, string)
        print(item)

    #     print(key)
    #     min_timems = data[0]['timems']
    #     values = []
    #     for i in range(3):
    #         cur = data[i]
    #         maxdiff = abs(min_timems - cur['timems']) / min_timems
    #         if maxdiff <= 0.05:
    #             string = "{}x{}_{}x{}x{}_{}".format(cur['WG_M'], cur['WG_N'], cur['SG_M'], cur['SG_N'], cur['SG_K'], cur['SLM_KS'])
    #             print(string)
    #             values.append(string)
    #             if policy_count.get(string) is None:
    #                 policy_count[string] = 1
    #             else:
    #                 policy_count[string] += 1
    #     mnk2policy[key] = values
    # sorted_policy = sorted(policy_count, key=policy_count.get, reverse=True)
    # print('\n===============  RESULTS ===============\n')
    # for policy in sorted_policy:
    #     print(policy)
    #     for key in mnk2policy:
    #         if policy in mnk2policy[key]:
    #             print(key)
    #             mnk2policy[key] = []
    #     print(' ')


if __name__ == '__main__':
    main()
