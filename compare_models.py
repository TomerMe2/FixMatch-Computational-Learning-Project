from scipy.stats import f_oneway
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import chain


if __name__ == '__main__':
    fixmatch_acc = pd.read_csv('metrics_log_fixmatch.csv')['acc']
    ratio_fixmatch_acc = pd.read_csv('metrics_log_ratio_fix_match.csv')['acc']
    label_spreading_acc = pd.read_csv('metrics_log_Label_spreading.csv')['acc']

    stat, p_val = f_oneway(fixmatch_acc, ratio_fixmatch_acc, label_spreading_acc)
    if p_val < 0.05:
        print('ANOVA found that there are algorithms that are different from each other significantly.')
        print('Doing Tukey Post Hoc test to determine which one is different.')

        m_comp = pairwise_tukeyhsd(endog=list(chain(fixmatch_acc, ratio_fixmatch_acc, label_spreading_acc)),
                                   groups=['FixMatch'] * len(fixmatch_acc) +
                                          ['RatioFixMatch'] * len(ratio_fixmatch_acc) +
                                          ['LabelSpreading'] * len(label_spreading_acc), alpha=0.05)
        print(m_comp)

    else:
        print("ANOVA didn't find that there are algorithms that are different from each other significantly")