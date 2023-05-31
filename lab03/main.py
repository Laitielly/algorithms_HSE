from prepare_report import knapsack_report, salesman_report
from data.get_branches import get_salesman, get_knapsack


capacity, profits, weights, optimal = get_knapsack('data/benchmarks/')
graph = get_salesman('data/output_graphs.npy')

# print(knapsack_report(capacity, weights, profits))
print(salesman_report(graph))

# capacity = 6404180
# weights = [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111, 323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684]
# profit = [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261]

# capacity = 165
# weights = [23,31,29,44,53,38,63,85,89,82]
# profit = [92,57,49,68,60,43,67,84,87,72]

# a = Knapsack(weights, profit, capacity)
# print(a.calculate(100, 10000, 10))

# matrix = [[0,5,7,1],[4,0,8,9],[1,5,0,6],[3,6,1,0]]

# matrix = [[  0., 633., 257.,  91., 412., 150.,  80., 134., 259., 505., 353.,
#          324.,  70., 211., 268., 246., 121.],
#         [633.,   0., 390., 661., 227., 488., 572., 530., 555., 289., 282.,
#          638., 567., 466., 420., 745., 518.],
#         [257., 390.,   0., 228., 169., 112., 196., 154., 372., 262., 110.,
#          437., 191.,  74.,  53., 472., 142.],
#         [ 91., 661., 228.,   0., 383., 120.,  77., 105., 175., 476., 324.,
#          240.,  27., 182., 239., 237.,  84.],
#         [412., 227., 169., 383.,   0., 267., 351., 309., 338., 196.,  61.,
#          421., 346., 243., 199., 528., 297.],
#         [150., 488., 112., 120., 267.,   0.,  63.,  34., 264., 360., 208.,
#          329.,  83., 105., 123., 364.,  35.],
#         [ 80., 572., 196.,  77., 351.,  63.,   0.,  29., 232., 444., 292.,
#          297.,  47., 150., 207., 332.,  29.],
#         [134., 530., 154., 105., 309.,  34.,  29.,   0., 249., 402., 250.,
#          314.,  68., 108., 165., 349.,  36.],
#         [259., 555., 372., 175., 338., 264., 232., 249.,   0., 495., 352.,
#           95., 189., 326., 383., 202., 236.],
#         [505., 289., 262., 476., 196., 360., 444., 402., 495.,   0., 154.,
#          578., 439., 336., 240., 685., 390.],
#         [353., 282., 110., 324.,  61., 208., 292., 250., 352., 154.,   0.,
#          435., 287., 184., 140., 542., 238.],
#         [324., 638., 437., 240., 421., 329., 297., 314.,  95., 578., 435.,
#            0., 254., 391., 448., 157., 301.],
#         [ 70., 567., 191.,  27., 346.,  83.,  47.,  68., 189., 439., 287.,
#          254.,   0., 145., 202., 289.,  55.],
#         [211., 466.,  74., 182., 243., 105., 150., 108., 326., 336., 184.,
#          391., 145.,   0.,  57., 426.,  96.],
#         [268., 420.,  53., 239., 199., 123., 207., 165., 383., 240., 140.,
#          448., 202.,  57.,   0., 483., 153.],
#         [246., 745., 472., 237., 528., 364., 332., 349., 202., 685., 542.,
#          157., 289., 426., 483.,   0., 336.],
#         [121., 518., 142.,  84., 297.,  35.,  29.,  36., 236., 390., 238.,
#          301.,  55.,  96., 153., 336.,   0.]]
#
# a = TravellingSalesman(matrix)
# print(a.calculate(10000, 10, 1000))

# print(get_knapsack('data/benchmarks/'))
# print(get_salesman('data/output_graphs.npy'))