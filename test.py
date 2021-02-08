import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

font = {'size': 6}

matplotlib.rc('font', **font)


x = [[3, 3, 3, 3, 2, 1, 3, 1], [20, 20, 20, 20, 17, 8, 20, 7], [20, 20, 20, 20, 17, 7, 20, 6], [13, 13, 13, 13, 11, 7,
                                                                                                13, 5], [0, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 2, 2, 1, 0], [20, 20, 20, 20, 16, 6, 20, 8], [0, 0, 0, 0, 0, 0, 0, 0]]
y = [[3, 3, 3, 3, 3, 3, 3, 3], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [13, 13, 13, 13, 13, 13, 13, 13], [
    20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20]]

z = [[0 for _ in range(8)]
     for _ in range(8)]

for i in range(8):
    for j in range(8):
        z[i][j] = x[i][j]/y[i][j]
z = z[::-1]
print(z)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.imshow(z, interpolation='none', cmap=cm.Greys_r)

x_label_list = ['L2_FGSM', 'L2_DFool', 'L2_C&W', 'DDN',
                'Linf_BI', 'Linf_FGSM', 'Linf_Dfool', 'Linf_PGD']
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax1.set_xticklabels(x_label_list)
ax1.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])
ax1.set_yticklabels(x_label_list)

plt.title('Recover Rate', fontsize=18)
plt.xlabel('Attack')
plt.ylabel('Recover')

plt.show()
