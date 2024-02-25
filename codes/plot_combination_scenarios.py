import matplotlib.pyplot as plt
import pandas as pd
from semantic_encoding import plot_res
import numpy as np
import matplotx
import seaborn as sns

clr = ['', 'g', 'b', 'm']
mark = ['o', 'o', 'o', 'o']
dt = pd.read_excel("discussion/combination.xlsx")
grps = dt.groupby(by="cmName")
with plt.style.context(matplotx.styles.pitaya_smoothie['light']):
    plt.figure()
    identity_line = np.linspace(0.2, 2.0)
    plt.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=1.0)

    for i, (ix, g) in enumerate(grps):
        print(ix)
        if ix == "Install a combination of shoulder rumble stripes, shoulder " \
                 "widening (from 0 to 2 ft) and resurface pavement":
            ix = 'Combination'
        if ix == 'Combination':
            cg = g
        else:
            plt.scatter(g['cmf_preds'], g['cmf'], alpha=0.8, marker=mark[i], c=clr[i], s=8, label=ix)

    plt.scatter(cg['cmf_preds'], cg['cmf'], alpha=1.0, marker='^', c='darkorange', s=20, label='Combination')

    plt.xlabel("Reported CMF values")
    plt.ylabel("CMF Predictions")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./aap_revision/customized_scatter_revised.pdf', format='pdf')
    plt.show()

    # plt.figure()
    # sns.violinplot(x="cmName", y="cmf", data=dt, color="0.8")
    # sns.stripplot(x="cmName", y="cmf", data=dt, jitter=True, zorder=1)
    # plt.show()

print('...')
