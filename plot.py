import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

matplotlib.rc("figure", dpi=250)

single = np.zeros((10,30))
fedavg = np.zeros((10,100))
meta = np.zeros((10,20))
our = np.zeros((10,100))

for n in range(10):
    node = str(n)

    with open('experiment_fedavg/'+node+'/metric.txt') as f:
        lines = f.readlines()

    ps_fedavg = [float(i) for i in lines[1][:-1].split()]
    fedavg[n] = np.array(ps_fedavg)

    with open('experiment_single/'+node+'/single_inr_metric.txt') as f:
        lines2 = f.readlines()

    ps_single = [float(i) for i in lines2[1][:-1].split()]
    single[n] = np.array(ps_single)

    with open('experiment_our/'+node+'/metric.txt') as f:
        lines2 = f.readlines()

    ps_our_new = [float(i) for i in lines2[1][:-1].split()]
    our[n] = np.array(ps_our_new)

    with open('experiment_meta/'+node+'/metrics.txt') as f:
        lines2 = f.readlines()

    ps_meta = [float(i) for i in lines2[1][:-1].split()]
    meta[n] = np.array(ps_meta)

mean_single = np.mean(single, axis=0)
se_single = scipy.stats.sem(single, axis=0)
mean_fedavg = np.mean(fedavg, axis=0)
se_fedavg = scipy.stats.sem(fedavg, axis=0)
mean_meta = np.mean(meta, axis=0)
se_meta = scipy.stats.sem(meta, axis=0)
mean_our = np.mean(our, axis=0)
se_our = scipy.stats.sem(our, axis=0)

fig = plt.figure(figsize=(8, 4.8))
plt.plot(range(10200, 30000+300, 300), mean_our[33:], label='INR-Bayes')
plt.fill_between(range(10200, 30000+300, 300), mean_our[33:] - se_our[33:], mean_our[33:] + se_our[33:], alpha=0.2)
plt.plot([x for x in range(11000, 30000+1000, 1000)], mean_meta, label='MAML')
plt.fill_between([x for x in range(11000, 30000+1000, 1000)], mean_meta - se_meta, mean_meta + se_meta, alpha=0.2)
plt.plot(range(10200, 30000+300, 300), mean_fedavg[33:], label='FedAvg')
plt.fill_between(range(10200, 30000+300, 300), mean_fedavg[33:] - se_fedavg[33:], mean_fedavg[33:] + se_fedavg[33:], alpha=0.2)
plt.plot(range(11000, 30000+1000, 1000), mean_single[10:], label='SingleINR')
plt.fill_between(range(11000, 30000+1000, 1000), mean_single[10:] - se_single[10:], mean_single[10:] + se_single[10:], alpha=0.2)

plt.xticks(range(10000, 40000, 10000), ['10k', '20k', '30k'])

plt.legend(loc='lower right', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('Total Iterations', fontsize=20)
plt.ylabel('PSNR / dB', fontsize=20)
plt.tight_layout()
plt.savefig('adapt_errorbar.pdf')




# import os 
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from imageio.v2 import imread
# import matplotlib.pyplot as plt
# import matplotlib
# import scipy
# matplotlib.rc("figure", dpi=250)

# single = np.zeros((10,60))
# fedavg = np.zeros((10,200))
# meta = np.zeros((10,50))
# our = np.zeros((10,200))

# for n in range(10):
#     node = str(n)

#     # with open('decathlon_fedavg/1/'+node+'/metric.txt') as f:
#     with open('decathlon_fedavg_1wadapt/'+node+'/metric.txt') as f:

#         lines = f.readlines()

#     ps_longer = [float(i) for i in lines[1][:-1].split()]
#     fedavg[n] = np.array(ps_longer)

#     with open('decathlon_single/1/'+node+'/single_inr_metric.txt') as f:
#         lines2 = f.readlines()

#     ps_single = [float(i) for i in lines2[1][:-1].split()]
#     single[n] = np.array(ps_single)

#     with open('self_adapt_our/1/'+node+'/metric.txt') as f:
#         lines2 = f.readlines()

#     ps_our_new = [float(i) for i in lines2[1][:-1].split()]
#     our[n] = np.array(ps_our_new)

#     # with open('decathlon_meta/1/'+node+'/metrics.txt') as f:
#     #     lines2 = f.readlines()

#     # ps_meta_old = [float(i) for i in lines2[1][:-1].split()]

#     with open('decathlon_meta_new/1/'+node+'/metrics.txt') as f:
#         lines2 = f.readlines()

#     ps_meta = [float(i) for i in lines2[1][:-1].split()]
#     meta[n] = np.array(ps_meta)

# mean_single = np.mean(single, axis=0)
# se_single = scipy.stats.sem(single, axis=0)
# mean_fedavg = np.mean(fedavg, axis=0)
# se_fedavg = scipy.stats.sem(fedavg, axis=0)
# mean_meta = np.mean(meta, axis=0)
# se_meta = scipy.stats.sem(meta, axis=0)
# mean_our = np.mean(our, axis=0)
# se_our = scipy.stats.sem(our, axis=0)

# fig = plt.figure(figsize=(8, 4.8))
# plt.plot(range(10200, 60000+300, 300), mean_our[33:], label='INR-Bayes')
# plt.fill_between(range(10200, 60000+300, 300), mean_our[33:] - se_our[33:], mean_our[33:] + se_our[33:], alpha=0.2)
# plt.plot([x for x in range(11000, 60000+1000, 1000)], mean_meta, label='MAML')
# plt.fill_between([x for x in range(11000, 60000+1000, 1000)], mean_meta - se_meta, mean_meta + se_meta, alpha=0.2)
# plt.plot(range(10200, 60000+300, 300), mean_fedavg[33:], label='FedAvg')
# plt.fill_between(range(10200, 60000+300, 300), mean_fedavg[33:] - se_fedavg[33:], mean_fedavg[33:] + se_fedavg[33:], alpha=0.2)
# plt.plot(range(11000, 60000+1000, 1000), mean_single[10:], label='SingleINR')
# plt.fill_between(range(11000, 60000+1000, 1000), mean_single[10:] - se_single[10:], mean_single[10:] + se_single[10:], alpha=0.2)

# plt.xticks(range(10000, 70000, 10000), ['10k', '20k', '30k', '40k', '50k', '60k'])

# plt.axvline(x=30000, color='r', linestyle='--')
# plt.text(30000, plt.ylim()[1]/3*2.5, '--> further adaptation ', horizontalalignment='left', verticalalignment='center', fontsize=18)

# # plt.axvline(x=10000, color='r', linestyle='--')
# # plt.text(10000, plt.ylim()[1]/3*1.2, '--> self-adapt meta ', horizontalalignment='left', verticalalignment='bottom', fontsize=14)


# plt.legend(loc='lower right', fontsize=16)
# plt.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('Total Iterations', fontsize=20)
# plt.ylabel('PSNR / dB', fontsize=20)
# plt.tight_layout()
# # plt.title('')
# plt.savefig('adapt_errorbar.pdf')