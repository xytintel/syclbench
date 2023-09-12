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
    check_call(['bash', 'build.sh', './gemm/hgemm_xetla.cpp'])
    datas = []
    for shape in shapes:
        m = shape[0]
        n = shape[1]
        k = shape[2]
        print("collecting: m={}, n={}, k={}".format(m, n, k))
        output = check_output(['./a.out', str(m), str(n), str(k)])
        output = output.decode('utf-8')
        datas.append(sort_out_policy(output))
    policy_count = {}
    mnk2policy = {}
    for data in datas:
        key = "{{{}, {}, {}}}".format(data[0]['m'], data[0]['n'], data[0]['k'])
        cur = data[0]
        string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(cur['WG_M'], cur['WG_N'], cur['SG_M'], cur['SG_N'], cur['SG_K'], cur['SLM_KS'])
        item = "{{{}, {}}},".format(key, string)
        print(item)


if __name__ == '__main__':
    main()
