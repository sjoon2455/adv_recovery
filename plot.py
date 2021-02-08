import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

font = {'size': 6}

matplotlib.rc('font', **font)


x1 = [[0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [0, 13, 13, 13, 11, 11,
                                                                                                     13, 11], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [0, 0, 0, 0, 0, 0, 0, 0]]
y1 = [[0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [13, 13, 13, 13, 13, 13,
                                                                                                     13, 13], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [0, 0, 0, 0, 0, 0, 0, 0]]
z1 = [[0 for _ in range(8)]
      for _ in range(8)]

x2 = [[0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [10, 13, 13, 13, 13, 13,
                                                                                                     13, 13], [2, 4, 4, 5, 6, 5, 4, 7], [0, 3, 2, 3, 4, 3, 2, 4], [20, 20, 20, 20, 20, 20, 20, 20], [1, 2, 2, 3, 5, 5, 3, 6]]
y2 = [[0, 0, 0, 0, 0, 0, 0, 0], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [13, 13, 13, 13, 13, 13,
                                                                                                     13, 13], [8, 8, 8, 8, 8, 8, 8, 8], [6, 6, 6, 6, 6, 6, 6, 6], [20, 20, 20, 20, 20, 20, 20, 20], [7, 7, 6, 7, 7, 8, 6, 7]]
z2 = [[0 for _ in range(8)]
      for _ in range(8)]


x3 = [[3, 3, 3, 3, 2, 1, 3, 1], [20, 20, 20, 20, 17, 8, 20, 7], [20, 20, 20, 20, 17, 7, 20, 6], [13, 13, 13, 13, 11, 7,
                                                                                                 13, 5], [0, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 2, 2, 1, 0], [20, 20, 20, 20, 16, 6, 20, 8], [0, 0, 0, 0, 0, 0, 0, 0]]
y3 = [[3, 3, 3, 3, 3, 3, 3, 3], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [13, 13, 13, 13, 13, 13, 13, 13], [
    20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20]]

z3 = [[0 for _ in range(8)]
      for _ in range(8)]

for i in range(8):
    for j in range(8):
        if y1[i][j] == 0:
            z1[i][j] == 1
        else:
            z1[i][j] = x1[i][j]/y1[i][j]
        if y2[i][j] == 0:
            z2[i][j] == 1
        else:
            z2[i][j] = x2[i][j]/y2[i][j]
        if y3[i][j] == 0:
            z3[i][j] == 1
        else:
            z3[i][j] = x3[i][j]/y3[i][j]
z1 = z1[::-1]
z2 = z2[::-1]
z3 = z3[::-1]

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
plt.title('Recover Rate')

ax1.imshow(z1, interpolation='none', cmap=cm.Greys_r)
ax2.imshow(z2, interpolation='none', cmap=cm.Greys_r)
ax3.imshow(z3, interpolation='none', cmap=cm.Greys_r)

x_label_list = ['L2_FGSM', 'L2_DFool', 'L2_C&W', 'DDN',
                'Linf_BI', 'Linf_FGSM', 'Linf_Dfool', 'Linf_PGD']
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax1.set_xticklabels(x_label_list)
ax1.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])
ax1.set_yticklabels(x_label_list)
ax1.set_xlabel('Attack')
ax1.set_ylabel('Recover')
ax1.set_title('Recover Rate(ε = 0.01)')

ax2.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax2.set_xticklabels(x_label_list)
ax2.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])
ax2.set_yticklabels(x_label_list)
ax2.set_xlabel('Attack')
ax2.set_ylabel('Recover')
ax2.set_title('Recover Rate(ε = 0.1)')

ax3.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
ax3.set_xticklabels(x_label_list)
ax3.set_yticks([7, 6, 5, 4, 3, 2, 1, 0])
ax3.set_yticklabels(x_label_list)
ax3.set_xlabel('Attack')
ax3.set_ylabel('Recover')
ax3.set_title('Recover Rate(ε = 0.8)')


plt.show()
