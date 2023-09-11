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


def main():
    shapes = get_all_shapes('./gemm/focus_shapes.txt')
    check_call(['bash', 'build.sh', './gemm/hgemm_xetla.cpp'])
    datas = []
    with open('hgemm.data', 'w') as f:
        for shape in shapes:
            m = shape[0]
            n = shape[1]
            k = shape[2]
            print("collecting: m={}, n={}, k={}".format(m, n, k))
            output = check_output(['./a.out', str(m), str(n), str(k)])
            output = output.decode('utf-8')
            f.write(output)


if __name__ == '__main__':
    main()
