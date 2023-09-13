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
        return num_ss, wg_eff, aspect_r


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


policy2id = {}
for policy in policies:
    string = "hgemm_policy::_{}x{}_{}x{}x{}_{}_true_".format(policy.wg_m, 
                    policy.wg_n, policy.sg_m, policy.sg_n, policy.sg_k, policy.slm_ks)
    policy2id[string] = policy.id


sample_string = """
{{1, 128, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 0
{{1, 128, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 1
{{1, 128, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 2
{{1, 128, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 3
{{1, 128, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 4
{{1, 128, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 5
{{1, 128, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 6
{{1, 128, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 7
{{1, 128, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 8
{{1, 128, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 9
{{1, 128, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 10
{{1, 128, 32768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 11
{{1, 192, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 12
{{1, 192, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 13
{{1, 192, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 14
{{1, 192, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 15
{{1, 192, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 16
{{1, 192, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 17
{{1, 192, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 18
{{1, 192, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 19
{{1, 192, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 20
{{1, 192, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 21
{{1, 192, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 22
{{1, 192, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 23
{{1, 256, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 24
{{1, 256, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 25
{{1, 256, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 26
{{1, 256, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 27
{{1, 256, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 28
{{1, 256, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 29
{{1, 256, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 30
{{1, 256, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 31
{{1, 256, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 32
{{1, 256, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 33
{{1, 256, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 34
{{1, 256, 32768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 35
{{1, 384, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 36
{{1, 384, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 37
{{1, 384, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 38
{{1, 384, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 39
{{1, 384, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 40
{{1, 384, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 41
{{1, 384, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 42
{{1, 384, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 43
{{1, 384, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 44
{{1, 384, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 45
{{1, 384, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 46
{{1, 384, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 47
{{1, 512, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 48
{{1, 512, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 49
{{1, 512, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 50
{{1, 512, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 51
{{1, 512, 2048}, hgemm_policy::_8x64_8x16x32_8_true_}, // 52
{{1, 512, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 53
{{1, 512, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 54
{{1, 512, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 55
{{1, 512, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 56
{{1, 512, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 57
{{1, 512, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 58
{{1, 512, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 59
{{1, 768, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 60
{{1, 768, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 61
{{1, 768, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 62
{{1, 768, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 63
{{1, 768, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 64
{{1, 768, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 65
{{1, 768, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 66
{{1, 768, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 67
{{1, 768, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 68
{{1, 768, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 69
{{1, 768, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 70
{{1, 768, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 71
{{1, 1024, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 72
{{1, 1024, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 73
{{1, 1024, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 74
{{1, 1024, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 75
{{1, 1024, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 76
{{1, 1024, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 77
{{1, 1024, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 78
{{1, 1024, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 79
{{1, 1024, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 80
{{1, 1024, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 81
{{1, 1024, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 82
{{1, 1024, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 83
{{1, 1536, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 84
{{1, 1536, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 85
{{1, 1536, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 86
{{1, 1536, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 87
{{1, 1536, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 88
{{1, 1536, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 89
{{1, 1536, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 90
{{1, 1536, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 91
{{1, 1536, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 92
{{1, 1536, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 93
{{1, 1536, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 94
{{1, 1536, 32768}, hgemm_policy::_32x64_32x16x16_8_true_}, // 95
{{1, 2048, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 96
{{1, 2048, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 97
{{1, 2048, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 98
{{1, 2048, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 99
{{1, 2048, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 100
{{1, 2048, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 101
{{1, 2048, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 102
{{1, 2048, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 103
{{1, 2048, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 104
{{1, 2048, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 105
{{1, 2048, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 106
{{1, 2048, 32768}, hgemm_policy::_32x64_32x16x16_8_true_}, // 107
{{1, 3072, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 108
{{1, 3072, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 109
{{1, 3072, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 110
{{1, 3072, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 111
{{1, 3072, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 112
{{1, 3072, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 113
{{1, 3072, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 114
{{1, 3072, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 115
{{1, 3072, 8192}, hgemm_policy::_32x64_8x16x16_2_true_}, // 116
{{1, 3072, 12288}, hgemm_policy::_32x64_8x16x16_2_true_}, // 117
{{1, 3072, 16384}, hgemm_policy::_128x128_32x32x32_2_true_}, // 118
{{1, 3072, 32768}, hgemm_policy::_128x128_32x32x32_2_true_}, // 119
{{1, 4096, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 120
{{1, 4096, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 121
{{1, 4096, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 122
{{1, 4096, 1536}, hgemm_policy::_32x64_32x16x16_8_true_}, // 123
{{1, 4096, 2048}, hgemm_policy::_32x64_8x16x16_2_true_}, // 124
{{1, 4096, 3072}, hgemm_policy::_128x64_16x16x64_1_true_}, // 125
{{1, 4096, 4096}, hgemm_policy::_128x64_16x16x64_1_true_}, // 126
{{1, 4096, 6144}, hgemm_policy::_128x64_16x16x64_1_true_}, // 127
{{1, 4096, 8192}, hgemm_policy::_128x64_16x16x64_1_true_}, // 128
{{1, 4096, 12288}, hgemm_policy::_128x64_16x16x64_1_true_}, // 129
{{1, 4096, 16384}, hgemm_policy::_128x64_16x16x64_1_true_}, // 130
{{1, 4096, 32768}, hgemm_policy::_128x128_16x32x64_1_true_}, // 131
{{1, 6144, 512}, hgemm_policy::_8x128_8x16x16_2_true_}, // 132
{{1, 6144, 768}, hgemm_policy::_8x128_8x16x16_2_true_}, // 133
{{1, 6144, 1024}, hgemm_policy::_32x128_32x16x16_4_true_}, // 134
{{1, 6144, 1536}, hgemm_policy::_8x128_8x16x16_2_true_}, // 135
{{1, 6144, 2048}, hgemm_policy::_8x128_8x16x16_2_true_}, // 136
{{1, 6144, 3072}, hgemm_policy::_8x128_8x16x16_2_true_}, // 137
{{1, 6144, 4096}, hgemm_policy::_8x128_8x16x16_2_true_}, // 138
{{1, 6144, 6144}, hgemm_policy::_8x128_8x16x16_2_true_}, // 139
{{1, 6144, 8192}, hgemm_policy::_8x128_8x16x16_2_true_}, // 140
{{1, 6144, 12288}, hgemm_policy::_128x128_16x32x64_1_true_}, // 141
{{1, 6144, 16384}, hgemm_policy::_8x128_8x16x16_2_true_}, // 142
{{1, 6144, 32768}, hgemm_policy::_8x128_8x16x16_2_true_}, // 143
{{1, 8192, 512}, hgemm_policy::_128x128_16x32x64_1_true_}, // 144
{{1, 8192, 768}, hgemm_policy::_128x128_16x32x64_1_true_}, // 145
{{1, 8192, 1024}, hgemm_policy::_8x128_8x16x32_4_true_}, // 146
{{1, 8192, 1536}, hgemm_policy::_128x64_16x16x64_1_true_}, // 147
{{1, 8192, 2048}, hgemm_policy::_128x64_16x16x64_1_true_}, // 148
{{1, 8192, 3072}, hgemm_policy::_128x256_32x32x16_1_true_}, // 149
{{1, 8192, 4096}, hgemm_policy::_8x256_8x16x16_2_true_}, // 150
{{1, 8192, 6144}, hgemm_policy::_128x256_32x32x16_1_true_}, // 151
{{1, 8192, 8192}, hgemm_policy::_8x256_8x16x16_2_true_}, // 152
{{1, 8192, 12288}, hgemm_policy::_128x256_32x32x16_1_true_}, // 153
{{1, 8192, 16384}, hgemm_policy::_128x256_32x32x16_1_true_}, // 154
{{1, 8192, 32768}, hgemm_policy::_128x256_32x32x16_1_true_}, // 155
{{1, 12288, 512}, hgemm_policy::_16x256_8x16x16_1_true_}, // 156
{{1, 12288, 768}, hgemm_policy::_128x256_32x32x16_1_true_}, // 157
{{1, 12288, 1024}, hgemm_policy::_16x256_8x16x16_1_true_}, // 158
{{1, 12288, 1536}, hgemm_policy::_16x256_8x16x16_1_true_}, // 159
{{1, 12288, 2048}, hgemm_policy::_128x256_32x32x16_1_true_}, // 160
{{1, 12288, 3072}, hgemm_policy::_128x256_64x16x16_1_true_}, // 161
{{1, 12288, 4096}, hgemm_policy::_256x256_32x64x16_1_true_}, // 162
{{1, 12288, 6144}, hgemm_policy::_256x256_32x64x16_1_true_}, // 163
{{1, 12288, 8192}, hgemm_policy::_128x256_64x16x16_1_true_}, // 164
{{1, 12288, 12288}, hgemm_policy::_256x256_32x64x16_1_true_}, // 165
{{1, 12288, 16384}, hgemm_policy::_256x256_64x32x16_1_true_}, // 166
{{1, 12288, 32768}, hgemm_policy::_128x256_64x16x16_1_true_}, // 167
{{1, 16384, 512}, hgemm_policy::_16x256_8x16x16_1_true_}, // 168
{{1, 16384, 768}, hgemm_policy::_128x256_64x16x16_1_true_}, // 169
{{1, 16384, 1024}, hgemm_policy::_8x512_8x16x16_1_true_}, // 170
{{1, 16384, 1536}, hgemm_policy::_16x512_16x16x16_1_true_}, // 171
{{1, 16384, 2048}, hgemm_policy::_8x512_8x16x16_1_true_}, // 172
{{1, 16384, 3072}, hgemm_policy::_8x512_8x16x16_1_true_}, // 173
{{1, 16384, 4096}, hgemm_policy::_8x512_8x16x16_1_true_}, // 174
{{1, 16384, 6144}, hgemm_policy::_8x512_8x16x16_1_true_}, // 175
{{1, 16384, 8192}, hgemm_policy::_8x512_8x16x16_1_true_}, // 176
{{1, 16384, 12288}, hgemm_policy::_8x512_8x16x16_1_true_}, // 177
{{1, 16384, 16384}, hgemm_policy::_8x512_8x16x16_1_true_}, // 178
{{1, 16384, 32768}, hgemm_policy::_8x512_8x16x16_1_true_}, // 179
{{1, 32768, 512}, hgemm_policy::_128x256_64x16x16_1_true_}, // 180
{{1, 32768, 768}, hgemm_policy::_128x256_32x32x16_1_true_}, // 181
{{1, 32768, 1024}, hgemm_policy::_128x256_32x32x16_1_true_}, // 182
{{1, 32768, 1536}, hgemm_policy::_256x256_32x64x16_1_true_}, // 183
{{1, 32768, 2048}, hgemm_policy::_256x256_32x64x16_1_true_}, // 184
{{1, 32768, 3072}, hgemm_policy::_256x256_32x64x16_1_true_}, // 185
{{1, 32768, 4096}, hgemm_policy::_256x256_32x64x16_1_true_}, // 186
{{1, 32768, 6144}, hgemm_policy::_256x256_32x64x16_1_true_}, // 187
{{1, 32768, 8192}, hgemm_policy::_256x256_32x64x16_1_true_}, // 188
{{1, 32768, 12288}, hgemm_policy::_256x256_32x64x16_1_true_}, // 189
{{1, 32768, 16384}, hgemm_policy::_128x256_32x32x16_1_true_}, // 190
{{1, 32768, 32768}, hgemm_policy::_256x256_32x64x16_1_true_}, // 191
{{4, 128, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 192
{{4, 128, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 193
{{4, 128, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 194
{{4, 128, 1536}, hgemm_policy::_32x64_32x16x16_8_true_}, // 195
{{4, 128, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 196
{{4, 128, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 197
{{4, 128, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 198
{{4, 128, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 199
{{4, 128, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 200
{{4, 128, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 201
{{4, 128, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 202
{{4, 128, 32768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 203
{{4, 192, 512}, hgemm_policy::_8x64_8x16x32_8_true_}, // 204
{{4, 192, 768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 205
{{4, 192, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 206
{{4, 192, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 207
{{4, 192, 2048}, hgemm_policy::_32x64_32x16x16_8_true_}, // 208
{{4, 192, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 209
{{4, 192, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 210
{{4, 192, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 211
{{4, 192, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 212
{{4, 192, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 213
{{4, 192, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 214
{{4, 192, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 215
{{4, 256, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 216
{{4, 256, 768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 217
{{4, 256, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 218
{{4, 256, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 219
{{4, 256, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 220
{{4, 256, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 221
{{4, 256, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 222
{{4, 256, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 223
{{4, 256, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 224
{{4, 256, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 225
{{4, 256, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 226
{{4, 256, 32768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 227
{{4, 384, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 228
{{4, 384, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 229
{{4, 384, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 230
{{4, 384, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 231
{{4, 384, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 232
{{4, 384, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 233
{{4, 384, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 234
{{4, 384, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 235
{{4, 384, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 236
{{4, 384, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 237
{{4, 384, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 238
{{4, 384, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 239
{{4, 512, 512}, hgemm_policy::_8x64_8x16x32_8_true_}, // 240
{{4, 512, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 241
{{4, 512, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 242
{{4, 512, 1536}, hgemm_policy::_8x64_8x16x32_8_true_}, // 243
{{4, 512, 2048}, hgemm_policy::_8x64_8x16x32_8_true_}, // 244
{{4, 512, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 245
{{4, 512, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 246
{{4, 512, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 247
{{4, 512, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 248
{{4, 512, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 249
{{4, 512, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 250
{{4, 512, 32768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 251
{{4, 768, 512}, hgemm_policy::_8x128_8x16x32_4_true_}, // 252
{{4, 768, 768}, hgemm_policy::_8x128_8x16x32_4_true_}, // 253
{{4, 768, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 254
{{4, 768, 1536}, hgemm_policy::_8x64_8x16x32_8_true_}, // 255
{{4, 768, 2048}, hgemm_policy::_32x64_32x16x16_8_true_}, // 256
{{4, 768, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 257
{{4, 768, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 258
{{4, 768, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 259
{{4, 768, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 260
{{4, 768, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 261
{{4, 768, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 262
{{4, 768, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 263
{{4, 1024, 512}, hgemm_policy::_16x64_16x16x16_8_true_}, // 264
{{4, 1024, 768}, hgemm_policy::_8x128_8x16x32_4_true_}, // 265
{{4, 1024, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 266
{{4, 1024, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 267
{{4, 1024, 2048}, hgemm_policy::_8x64_8x16x32_8_true_}, // 268
{{4, 1024, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 269
{{4, 1024, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 270
{{4, 1024, 6144}, hgemm_policy::_16x64_16x16x16_8_true_}, // 271
{{4, 1024, 8192}, hgemm_policy::_16x64_16x16x16_8_true_}, // 272
{{4, 1024, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 273
{{4, 1024, 16384}, hgemm_policy::_16x64_16x16x16_8_true_}, // 274
{{4, 1024, 32768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 275
{{4, 1536, 512}, hgemm_policy::_8x128_8x16x32_4_true_}, // 276
{{4, 1536, 768}, hgemm_policy::_8x64_8x16x32_8_true_}, // 277
{{4, 1536, 1024}, hgemm_policy::_8x64_8x16x32_8_true_}, // 278
{{4, 1536, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 279
{{4, 1536, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 280
{{4, 1536, 3072}, hgemm_policy::_16x64_16x16x16_8_true_}, // 281
{{4, 1536, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 282
{{4, 1536, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 283
{{4, 1536, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 284
{{4, 1536, 12288}, hgemm_policy::_8x64_8x16x32_8_true_}, // 285
{{4, 1536, 16384}, hgemm_policy::_8x64_8x16x32_8_true_}, // 286
{{4, 1536, 32768}, hgemm_policy::_32x64_32x16x16_8_true_}, // 287
{{4, 2048, 512}, hgemm_policy::_8x64_8x16x32_8_true_}, // 288
{{4, 2048, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 289
{{4, 2048, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 290
{{4, 2048, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 291
{{4, 2048, 2048}, hgemm_policy::_32x64_32x16x16_8_true_}, // 292
{{4, 2048, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 293
{{4, 2048, 4096}, hgemm_policy::_8x64_8x16x32_8_true_}, // 294
{{4, 2048, 6144}, hgemm_policy::_32x64_32x16x16_8_true_}, // 295
{{4, 2048, 8192}, hgemm_policy::_8x64_8x16x32_8_true_}, // 296
{{4, 2048, 12288}, hgemm_policy::_16x64_16x16x16_8_true_}, // 297
{{4, 2048, 16384}, hgemm_policy::_32x64_32x16x16_8_true_}, // 298
{{4, 2048, 32768}, hgemm_policy::_32x64_32x16x16_8_true_}, // 299
{{4, 3072, 512}, hgemm_policy::_8x64_8x16x32_8_true_}, // 300
{{4, 3072, 768}, hgemm_policy::_8x128_8x16x32_4_true_}, // 301
{{4, 3072, 1024}, hgemm_policy::_16x64_16x16x16_8_true_}, // 302
{{4, 3072, 1536}, hgemm_policy::_16x64_16x16x16_8_true_}, // 303
{{4, 3072, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 304
{{4, 3072, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 305
{{4, 3072, 4096}, hgemm_policy::_16x64_16x16x16_8_true_}, // 306
{{4, 3072, 6144}, hgemm_policy::_8x64_8x16x32_8_true_}, // 307
{{4, 3072, 8192}, hgemm_policy::_32x64_8x16x16_2_true_}, // 308
{{4, 3072, 12288}, hgemm_policy::_32x64_8x16x16_2_true_}, // 309
{{4, 3072, 16384}, hgemm_policy::_32x64_8x16x16_2_true_}, // 310
{{4, 3072, 32768}, hgemm_policy::_32x64_8x16x16_2_true_}, // 311
{{4, 4096, 512}, hgemm_policy::_8x64_8x16x32_8_true_}, // 312
{{4, 4096, 768}, hgemm_policy::_16x64_16x16x16_8_true_}, // 313
{{4, 4096, 1024}, hgemm_policy::_8x128_8x16x32_4_true_}, // 314
{{4, 4096, 1536}, hgemm_policy::_8x64_8x16x32_8_true_}, // 315
{{4, 4096, 2048}, hgemm_policy::_8x128_8x16x32_4_true_}, // 316
{{4, 4096, 3072}, hgemm_policy::_8x64_8x16x32_8_true_}, // 317
{{4, 4096, 4096}, hgemm_policy::_32x64_8x16x16_2_true_}, // 318
{{4, 4096, 6144}, hgemm_policy::_128x64_16x16x64_1_true_}, // 319
{{4, 4096, 8192}, hgemm_policy::_128x64_16x16x64_1_true_}, // 320
{{4, 4096, 12288}, hgemm_policy::_32x64_8x16x16_2_true_}, // 321
{{4, 4096, 16384}, hgemm_policy::_32x64_8x16x16_2_true_}, // 322
{{4, 4096, 32768}, hgemm_policy::_128x128_16x32x64_1_true_}, // 323
{{4, 6144, 512}, hgemm_policy::_16x256_8x16x16_1_true_}, // 324
{{4, 6144, 768}, hgemm_policy::_8x128_8x16x32_4_true_}, // 325
{{4, 6144, 1024}, hgemm_policy::_8x128_8x16x32_4_true_}, // 326
{{4, 6144, 1536}, hgemm_policy::_8x128_8x16x32_4_true_}, // 327
{{4, 6144, 2048}, hgemm_policy::_16x64_16x16x16_8_true_}, // 328
{{4, 6144, 3072}, hgemm_policy::_8x128_8x16x32_4_true_}, // 329
{{4, 6144, 4096}, hgemm_policy::_8x128_8x16x32_4_true_}, // 330
{{4, 6144, 6144}, hgemm_policy::_8x256_8x16x16_2_true_}, // 331
{{4, 6144, 8192}, hgemm_policy::_8x256_8x16x16_2_true_}, // 332
{{4, 6144, 12288}, hgemm_policy::_8x256_8x16x16_2_true_}, // 333
{{4, 6144, 16384}, hgemm_policy::_8x128_8x16x16_2_true_}, // 334
{{4, 6144, 32768}, hgemm_policy::_8x128_8x16x16_2_true_}, // 335
{{4, 8192, 512}, hgemm_policy::_8x128_8x16x32_4_true_}, // 336
{{4, 8192, 768}, hgemm_policy::_16x256_8x16x16_1_true_}, // 337
{{4, 8192, 1024}, hgemm_policy::_8x128_8x16x32_4_true_}, // 338
{{4, 8192, 1536}, hgemm_policy::_8x256_8x16x16_2_true_}, // 339
{{4, 8192, 2048}, hgemm_policy::_8x128_8x16x16_2_true_}, // 340
{{4, 8192, 3072}, hgemm_policy::_8x128_8x16x16_2_true_}, // 341
{{4, 8192, 4096}, hgemm_policy::_8x128_8x16x16_2_true_}, // 342
{{4, 8192, 6144}, hgemm_policy::_8x128_8x16x16_2_true_}, // 343
{{4, 8192, 8192}, hgemm_policy::_8x128_8x16x16_2_true_}, // 344
{{4, 8192, 12288}, hgemm_policy::_8x256_8x16x16_2_true_}, // 345
{{4, 8192, 16384}, hgemm_policy::_8x256_8x16x16_2_true_}, // 346
{{4, 8192, 32768}, hgemm_policy::_8x256_8x16x16_2_true_}, // 347
{{4, 12288, 512}, hgemm_policy::_16x256_8x16x16_1_true_}, // 348
{{4, 12288, 768}, hgemm_policy::_16x256_8x16x16_1_true_}, // 349
{{4, 12288, 1024}, hgemm_policy::_16x256_8x16x16_1_true_}, // 350
{{4, 12288, 1536}, hgemm_policy::_16x256_8x16x16_1_true_}, // 351
{{4, 12288, 2048}, hgemm_policy::_16x256_8x16x16_1_true_}, // 352
{{4, 12288, 3072}, hgemm_policy::_16x256_8x16x16_1_true_}, // 353
{{4, 12288, 4096}, hgemm_policy::_16x256_8x16x16_1_true_}, // 354
{{4, 12288, 6144}, hgemm_policy::_128x256_64x16x16_1_true_}, // 355
{{4, 12288, 8192}, hgemm_policy::_16x256_8x16x16_1_true_}, // 356
{{4, 12288, 12288}, hgemm_policy::_128x256_64x16x16_1_true_}, // 357
{{4, 12288, 16384}, hgemm_policy::_128x256_64x16x16_1_true_}, // 358
{{4, 12288, 32768}, hgemm_policy::_128x256_64x16x16_1_true_}, // 359
{{4, 16384, 512}, hgemm_policy::_8x512_8x16x16_1_true_}, // 360
{{4, 16384, 768}, hgemm_policy::_16x256_8x16x16_1_true_}, // 361
{{4, 16384, 1024}, hgemm_policy::_16x256_8x16x16_1_true_}, // 362
{{4, 16384, 1536}, hgemm_policy::_16x256_8x16x16_1_true_}, // 363
{{4, 16384, 2048}, hgemm_policy::_8x512_8x16x16_1_true_}, // 364
{{4, 16384, 3072}, hgemm_policy::_8x512_8x16x16_1_true_}, // 365
{{4, 16384, 4096}, hgemm_policy::_16x512_16x16x16_1_true_}, // 366
{{4, 16384, 6144}, hgemm_policy::_8x512_8x16x16_1_true_}, // 367
{{4, 16384, 8192}, hgemm_policy::_8x512_8x16x16_1_true_}, // 368
{{4, 16384, 12288}, hgemm_policy::_8x512_8x16x16_1_true_}, // 369
{{4, 16384, 16384}, hgemm_policy::_8x512_8x16x16_1_true_}, // 370
{{4, 16384, 32768}, hgemm_policy::_8x512_8x16x16_1_true_}, // 371
{{4, 32768, 512}, hgemm_policy::_8x512_8x16x16_1_true_}, // 372
{{4, 32768, 768}, hgemm_policy::_8x512_8x16x16_1_true_}, // 373
{{4, 32768, 1024}, hgemm_policy::_16x256_8x16x16_1_true_}, // 374
{{4, 32768, 1536}, hgemm_policy::_16x256_8x16x16_1_true_}, // 375
{{4, 32768, 2048}, hgemm_policy::_16x256_8x16x16_1_true_}, // 376
{{4, 32768, 3072}, hgemm_policy::_16x256_8x16x16_1_true_}, // 377
{{4, 32768, 4096}, hgemm_policy::_16x256_8x16x16_1_true_}, // 378
{{4, 32768, 6144}, hgemm_policy::_128x256_32x32x16_1_true_}, // 379
{{4, 32768, 8192}, hgemm_policy::_256x256_32x64x16_1_true_}, // 380
{{4, 32768, 12288}, hgemm_policy::_128x256_32x32x16_1_true_}, // 381
{{4, 32768, 16384}, hgemm_policy::_128x256_32x32x16_1_true_}, // 382
{{4, 32768, 32768}, hgemm_policy::_256x256_32x64x16_1_true_}, // 383
"""

mnk2_policy_id = {}
for line in sample_string.split('\n'):
    if line.startswith("{{"):
        mnk_start = line.find('{') + 2
        mnk_end = line.find('}')
        mnk = line[mnk_start:mnk_end].strip()
        policy_start = line.find('hgemm_policy')
        policy_end = line.rfind('}')
        policy = line[policy_start:policy_end].strip()
        mnk2_policy_id[mnk] = policy2id[policy]
