# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:14:38 2025

@author: piercetf
"""

import scipy
import polars as pl
import seaborn as sb
import arviz as az
from matplotlib import pyplot
import pymc as pm
import pytensor as pt
import numpy as np
import xarray
import matplotlib_venn as venn
import itertools
#from sklearn import metrics as skmetrics

def load_data():

    SI_RNA = "db1_norelabel.xlsx"
    
    rna_to_gene = {
        1: 'PHIP',
        2: 'MFSD10',
        3: 'GMPR2',
        4: 'HAGHL',
        5: 'NSRP1',
        6: 'NOTCH2',
        7: 'CHAF1A',
        8: 'LILRB1',
        9: 'CENPC',
        10: 'JUNB',
        11: 'LETM1',
        12: 'MINK1',
        13: 'UQCR10',
        14: 'CD38',
        15: 'NRAS',
        16: 'SAFB2',
        17: 'ACOX1',
        18: 'GPD2',
        19: 'RTN3',
        20: 'SFXN3',
        21: 'COX7A2',
        22: 'SLC2A5',
        23: 'RECK',
        24: 'ATP5J2',
        25: 'DCTN5',
        26: 'NOLC1',
        27: 'TOR1AIP1',
        28: 'GNAQ',
        29: 'SLC16A1',
        30: 'TIMM8A',
        31: 'CCDC174',
        32: 'NDUFA7',
        33: 'NDUFAF2',
        34: 'GPALPP1',
        35: 'ACOT8',
        36: 'AGT',
        37: 'ASH2L',
        38: 'LRRC59',
        39: 'PYCRL',
        40: 'PPP1R12C',
        41: 'FMNL3',
        'rl': 'siCtrl'
        }
    
    rna_to_gene2 = {}
    for key, val in rna_to_gene.items():
        key2 = str(key)
        if len(key2) < 2:
            key2 = f"0{key2}"
        rna_to_gene2[key2] = val
    
    table = pl.read_excel(SI_RNA)
    table = table.rename({'luminiscence': 'luminescence'})
    table = table.with_columns(time=pl.col('time').str.strip_chars('T').str.to_integer(base=10))
    table = table.with_columns(log_lum=pl.col('luminescence').log1p()/np.log(2))
    table = table.with_columns(gene_name=pl.col('siRNA').str.slice(-2,None).replace_strict(rna_to_gene2))
    table = table.with_columns(log1time = pl.col('time').log1p())
    
    table = table.filter(pl.col('luminescence').is_not_null())
    # cannot rely on these genes to be silenced, so analyzing them at all is misleading
    table = table.filter(pl.col('gene_name').eq('CCDC174').not_() & 
                         pl.col('gene_name').eq('PHIP').not_() &
                         pl.col('gene_name').eq('ATP5J2').not_())
    # DMSO gets introduced at time 1 same as the other drugs
    relabel_treat = table.select(
        treatment=pl.when(pl.col('time').eq(0)).then(pl.lit('untreated')).otherwise(pl.col('treatment'))
        )['treatment']
    treat_idx = table.columns.index('treatment')
    table.replace_column(treat_idx, relabel_treat)
    table = table.with_columns(gene_idnum=pl.col('gene_name').rank('dense').sub(1),
                               treat_idnum=pl.col('treatment').rank('dense').sub(1))
    return table


def frequentist_treatment_impair(table):
    anova_table = []
    dunnett_tables = []
    days = table['time'].unique().sort()
    for day in days:
        daytab = table.filter(pl.col('time').eq(day))
        daytreats = daytab.select(pl.col('treatment')).unique()
        if len(daytreats) == 1:
            continue
        for treat in daytreats['treatment']:
            daytreattab = daytab.filter(pl.col('treatment').eq(treat))
            by_gene = daytreattab.group_by(['gene_name']).agg(pl.col('log_lum'))
            not_silenced = by_gene.filter(pl.col('gene_name').eq('siCtrl'))
            silenced = by_gene.filter(pl.col('gene_name').eq('siCtrl').not_())
            intreat_anova = scipy.stats.f_oneway(*by_gene['log_lum'])
            intreat_dunnett = scipy.stats.dunnett(*silenced['log_lum'],
                                                  control=not_silenced['log_lum'][0],
                                                  alternative='greater')
            anova_table.append({'day': day, 
                                'treatment': treat,
                                'anova_statistic': intreat_anova.statistic,
                                'anova_pvalue': intreat_anova.pvalue})
            dunnett_tables.append({'day': [day] * len(silenced),
                                   'treatment': [treat] * len(silenced),
                                   'gene': silenced['gene_name'],
                                   'dunnett_statistic': intreat_dunnett.statistic,
                                   'dunnett_pvalue': intreat_dunnett.pvalue})
    
    anova_ptable = pl.from_dicts(anova_table)
    dunnett_ptables = []
    for dtab in dunnett_tables:
        dunnett_ptable = pl.from_dict(dtab)
        dunnett_ptables.append(dunnett_ptable)
    dunnett_frame = pl.concat(dunnett_ptables)
    return anova_ptable, dunnett_frame
    


def frequentist_genediff(table):
    anova_table = []
    gene_tables = []
    for day in table['time'].unique().sort():
        daytab = table.filter(pl.col('time').eq(day))
        genetab = daytab.filter(pl.col('treatment').eq('untreated').or_(
            pl.col('treatment').eq('DMSO')))
        gene_only = genetab.group_by(['gene_name']).agg(pl.col('log_lum'))
        gene_ctrl = gene_only.filter(pl.col('gene_name').eq('siCtrl'))
        gene_test = gene_only.filter(pl.col('gene_name').eq('siCtrl').not_())
        gene_anova = scipy.stats.f_oneway(*gene_only['log_lum'])
        gene_dunnett = scipy.stats.dunnett(*gene_test['log_lum'],
                                           control=gene_ctrl['log_lum'][0],
                                           alternative='less')
        anova_table.append({'day': day, 
                            'anova_stat': gene_anova.statistic,
                            'anova_pvalue': gene_anova.pvalue})
        gene_tables.append({'day': [day] * len(gene_test),
                            'gene': gene_test['gene_name'],
                            'dunnett_pvalue': gene_dunnett.pvalue,
                            'dunnett_stat': gene_dunnett.statistic})
    anova_ptable = pl.from_dicts(anova_table)
    gene_ptables = []
    for table in gene_tables:
        gene_ptable = pl.from_dict(table)
        gene_ptables.append(gene_ptable)
    gene_ptable = pl.concat(gene_ptables)
    return anova_ptable, gene_ptable


def frequentist_analysis(table):
    # general goal: identify genes which influence survival
    gene_anovas, gene_table = frequentist_genediff(table)
    # identify genes which are significantly affecting survival at more than 2 time points
    sustained_surv_relevant = (gene_table
                               .filter(
                                   pl.col('dunnett_pvalue').lt(0.05)
                                   )
                               .group_by(
                                   ['gene']
                                   )
                               .agg(
                                   pl.col('dunnett_pvalue').count()
                                   )
                               .filter(
                                   pl.col('dunnett_pvalue').gt(2)
                                   ))
    sustained_surv_relevant.write_csv("survival_relevant.csv")
    surv_genes = set(sustained_surv_relevant['gene'])
    sb.barplot(gene_anovas, x='day', y='anova_pvalue')
    pyplot.title("gene anova by day")
    pyplot.savefig("survival_day_anova.svg", bbox_inches="tight")
    pyplot.show()
    cat = sb.catplot(gene_table, x='gene', y='dunnett_pvalue', col='day', kind='bar')
    cat.set_xticklabels(rotation=90)
    # bold survival relevant genes as previously identified
    for ax in cat.axes.flat:
        # hold tick position constant
        ax_ticks = ax.get_xticks()
        ax.set_xticks(ax_ticks)
        # get the ax labels
        ax_labels = ax.get_xticklabels()
        for ticklabel in ax_labels:
            ticktext = ticklabel.get_text()
            if ticktext in surv_genes:
                ticklabel.set_fontweight('bold')
        ax.axhline(0.05, color='red')
    pyplot.savefig("survival_daily_dunnett.svg", bbox_inches="tight")
    pyplot.show()
    
    impair_anova, impair_dunnett = frequentist_treatment_impair(table)
    
    drug_relevant = (impair_dunnett
                     .filter(pl.col('dunnett_pvalue').lt(0.05))
                     .group_by(['treatment', 'gene'])
                     .agg(pl.col('dunnett_pvalue').count())
                     .filter(pl.col('dunnett_pvalue').gt(1))
                     )
    
    drug_relevant.write_csv("drug_relevant.csv")
    
    sb.barplot(impair_anova, x='day', y='anova_pvalue', hue='treatment')
    pyplot.title("treatment anova by day")
    pyplot.savefig("treatment_daily_anova.svg", bbox_inches="tight")
    pyplot.show()
    
    cat = sb.catplot(impair_dunnett, 
                     x='gene', 
                     y='dunnett_pvalue', 
                     col='day', 
                     row='treatment',
                     kind='bar',
                     sharex=False)
    #cat.set_xticklabels(rotation=90)
    cat.tick_params(axis='x', which='both', rotation=90)
    for ax in cat.axes.flat:
        ax.axhline(0.05, color='red')
    for i,row_name in enumerate(cat.row_names):
        relevant_genes = drug_relevant.filter(pl.col('treatment').eq(row_name))['gene']
        drug_axes = cat.axes[i]
        for ax in drug_axes:
            for text in ax.get_xticklabels():
                if text.get_text() in relevant_genes:
                    text.set_fontweight('bold')
    cat.fig.tight_layout()
    pyplot.savefig("treatment_daily_dunnett.svg", bbox_inches="tight")
    pyplot.show()
    
    sustained_fisetin_relevant = drug_relevant.filter(pl.col('treatment').eq('Fisetin'))
    fisetin_genes = set(sustained_fisetin_relevant['gene'])
    sustained_quercetin_relevant = drug_relevant.filter(pl.col('treatment').eq('Quercetin'))
    quercetin_genes = set(sustained_quercetin_relevant['gene'])
    sustained_dmso_relevant = drug_relevant.filter(pl.col('treatment').eq('DMSO'))
    dmso_genes = set(sustained_dmso_relevant['gene'])
    
    surv_only = surv_genes - (fisetin_genes | quercetin_genes)
    surv_only_label = '\n'.join(map(lambda pair: '; '.join(pair),
                                    itertools.batched(surv_only, 2)
                                    )
                                )
    
    fis_only = fisetin_genes - (surv_genes | quercetin_genes)
    fisetin_only_label = '\n'.join(map(lambda pair: '; '.join(pair),
                                       itertools.batched(fis_only, 2)
                                       )
                                   )
    
    fis_and_quer = (fisetin_genes & quercetin_genes) - surv_genes
    fisandquer_label = '\n'.join(map(lambda pair: '; '.join(pair),
                                     itertools.batched(fis_and_quer, 2)
                                     )
                                 )
    
    all_test_genes = set(table['gene_name'])
    null_genes = all_test_genes - (surv_genes | fisetin_genes | quercetin_genes | dmso_genes | {'siCtrl'})
    
    fig, ax = pyplot.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(15)
    #pyplot.figure(fig, figsize=(10,10))
    v_diag = venn.venn3((surv_genes, fisetin_genes, quercetin_genes),
               ['Survival', 'Fisetin', 'Quercetin'],
               ax=ax)
    v_diag.get_label_by_id('100').set_text(surv_only_label)
    v_diag.get_label_by_id('010').set_text(fisetin_only_label)
    v_diag.get_label_by_id('011').set_text(fisandquer_label)
    
    v_diag.get_label_by_id('A').set_fontweight('bold')
    v_diag.get_label_by_id('B').set_fontweight('bold')
    v_diag.get_label_by_id('C').set_fontweight('bold')
    
    pyplot.text(0.5, 
                0.0,
                "No Detection\n" + '; '.join(null_genes), 
                transform=ax.transAxes,
                horizontalalignment='center')
    
    pyplot.text(0.0,
                0.0,
                "DMSO\n" + '; '.join(dmso_genes),
                transform=ax.transAxes,
                horizontalalignment='left')
    
    pyplot.text(1.0,
                0.0,
                "Not Testable\nPHIP; ATP5J2; CCDC174",
                transform=ax.transAxes,
                horizontalalignment='center')
    
    ax.set_title("siRNA Experiment Analysis Results Summary", 
                 fontweight='bold',
                 fontsize=20)
    pyplot.savefig("frequentist_analysis_summary.svg", bbox_inches="tight")
    pyplot.show()



if __name__ == '__main__':
    
    table = load_data()
    gene_lookup = table.select(pl.col('gene_idnum'),pl.col('gene_name')).unique()
    gene_lookup = gene_lookup.sort(pl.col('gene_idnum'))
    treat_lookup = table.select(pl.col('treat_idnum'), pl.col('treatment')).unique()
    treat_lookup = treat_lookup.sort(pl.col('treat_idnum'))
    coords = {'gene': gene_lookup['gene_name'], 
              'treatment': treat_lookup['treatment'],
              'placeholder': ['placeholder'],
              'day': table['time'].unique().sort()}
    
    viewcontroltab = table.filter(pl.col('gene_name').eq('siCtrl').and_(
        pl.col('treatment').eq('DMSO').or_(pl.col('treatment').eq('untreated'))
        ))
    
    sb.regplot(viewcontroltab,
                   x='time',
                   y='luminescence')
    pyplot.title("siCtrl + DMSO luminescence")
    pyplot.savefig('control_luminescence.svg')
    pyplot.show()
    table = table.with_columns(log_lum = pl.col('luminescence').log(base=2))
    frequentist_analysis(table)
    table2 = table.with_columns(std_lum = (pl.col('luminescence') - pl.col('luminescence').mean())/pl.col('luminescence').std())
    table_nodrug = (table2
                    .filter(
                        pl.col('treatment').eq('DMSO') |
                        pl.col('treatment').eq('untreated')
                        )
                    )
    sictrl_idx = gene_lookup.filter(pl.col('gene_name').eq('siCtrl'))['gene_idnum'][0]
    with pm.Model(coords=coords) as nodrug_model:
        std_lum = pm.Data('std_lum', table_nodrug['std_lum'])
        time = pm.Data('time', table_nodrug['time'])
        gene_idx = pm.Data('gene_idx', table_nodrug['gene_idnum'])
        baseline = pm.Normal('baseline', mu=0, sigma=1)
        gene_level = pm.Normal('gene_level',
                               mu=baseline,
                               sigma=1,
                               shape=(len(gene_lookup), 1),
                               dims=['gene', 'placeholder'])
        gene_day_level = pm.Normal('gene_day_level',
                                   mu=gene_level,
                                   sigma=1,
                                   shape=(len(gene_lookup), 4),
                                   dims=['gene', 'day'])
        control_level = pm.Deterministic('control_level',
                                         gene_day_level[sictrl_idx, :],
                                         dims=['day'])
        diff = pm.Deterministic('diff',
                               gene_day_level - control_level[np.newaxis, :],
                               dims=['gene', 'day'])
        tau = pm.Gamma('tau', alpha=100, beta=0.1)
        level = pm.Normal('level',
                          mu=gene_day_level[gene_idx, time],
                          tau=tau,
                          observed=std_lum)
    
    with nodrug_model:
        prior = pm.sample_prior_predictive()
        nodrug_itrace = pm.sample(draws=6000,
                                  nuts_sampler='blackjax',
                                  nuts={'target_accept': 0.95})
        post = pm.sample_posterior_predictive(nodrug_itrace)
        like = pm.compute_log_likelihood(nodrug_itrace)
    nodrug_itrace.extend(prior)
    nodrug_itrace.extend(post)
    
    nodrug_graph = nodrug_model.to_graphviz()
    nodrug_graph.render('nodrug_graph.gv', format='svg')
    
    # Confirming that the model makes sense
    az.plot_energy(nodrug_itrace)
    pyplot.title("Energy Diagnostic - Silencing Only")
    pyplot.savefig("energy_silence_only.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_ppc(nodrug_itrace)
    pyplot.title("Posterior Predictive Check - Silencing Only")
    pyplot.savefig("ppc_silence_only.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_loo_pit(nodrug_itrace, 'level')
    pyplot.title("LOO-PIT - Silencing Only")
    pyplot.savefig("loo_pit_silence_only.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_bpv(nodrug_itrace, 'p_value')
    pyplot.title("BPV - Silencing Only")
    pyplot.savefig("bpv_silence_only.svg", bbox_inches="tight")
    pyplot.show()
    
    tested_genes = gene_lookup.filter(pl.col('gene_name').eq('siCtrl').not_())['gene_name']
    control_levels = az.summary(nodrug_itrace, ['control_level'])
    az.plot_posterior(nodrug_itrace,
                      ['diff'],
                      coords={'day': 0, 
                              'gene': tested_genes},
                      ref_val=0,
                      grid=(4,10))
    pyplot.show()
    az.plot_posterior(nodrug_itrace,
                      ['diff'],
                      coords={'day': 1, 
                              'gene': tested_genes},
                      ref_val=0,
                      grid=(4,10))
    pyplot.show()
    az.plot_posterior(nodrug_itrace,
                      ['diff'],
                      coords={'day': 2, 
                              'gene': tested_genes},
                      ref_val=0,
                      grid=(4,10))
    pyplot.show()
    az.plot_posterior(nodrug_itrace,
                      ['diff'],
                      coords={'day': 3, 
                              'gene': tested_genes},
                      ref_val=0,
                      grid=(4,10))
    pyplot.show()
    # compute probability that the magnitude of viability drop under silencing
    # exceeds the baseline viability drop over time
    below_baseline_daily_probs = (nodrug_itrace.posterior['diff'] < 0).mean(('chain', 'draw'))
    # subsequent days occur in the context that the events of the previous days occurred,
    # that is they are not independent, and the probability obeys 
    # P(event of day N AND event of day N + 1) = P(event of day N) * P(event of day N + 1 | event of day N)
    # it is also the case that the probability we calculate for the subsequent days is
    # the conditional probability, not the marginal probability, so we can multiply out
    below_baseline_probs = below_baseline_daily_probs.prod(('day'))
    ax = sb.barplot(x=below_baseline_probs.gene, y=below_baseline_probs.data)
    pyplot.axhline(0.90, color='green')
    pyplot.xticks(rotation=90)
    pyplot.ylabel("Probability of sustained survival impact")
    pyplot.title("Survival Implication - Bayesian")
    xlabels = ax.get_xticklabels()
    for xlab in xlabels:
        gene = xlab.get_text()
        prob = float(below_baseline_probs.sel(gene=gene))
        if prob > 0.90:
            xlab.set_fontweight('bold')
    pyplot.savefig("semi_bayesian_survival.svg", bbox_inches="tight")
    pyplot.show()