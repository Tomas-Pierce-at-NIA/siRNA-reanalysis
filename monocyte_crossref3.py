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
import numpy as np
import matplotlib_venn as venn
import itertools
import re
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
    
    controls = ['PPP1R12C', 'FMNL3']
    
    surv_only = surv_genes - (fisetin_genes | quercetin_genes)
    surv_only_label = '\n'.join(map(lambda pair: '; '.join(pair),
                                    itertools.batched(filter(
                                        lambda name : name not in controls,
                                        surv_only
                                        ), 2)
                                    )
                                )
    if all(map(lambda c : c in surv_only, controls)):
        surv_only_label = '{}\n\nControls:\n{}'.format(surv_only_label, controls)
    
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
    
    bayesian_survival_implicated = set()
    
    for xlab in xlabels:
        gene = xlab.get_text()
        prob = float(below_baseline_probs.sel(gene=gene))
        if prob > 0.90:
            xlab.set_fontweight('bold')
            bayesian_survival_implicated.add(gene)
    pyplot.savefig("semi_bayesian_survival.svg", bbox_inches="tight")
    pyplot.show()
    
    nodrug_bayes_factors = {}
    figscale = 8
    for day in coords['day']:
        nodrug_bayes_factors[day] = {}
        fig, axes = pyplot.subplots(6, 7, figsize=(7 * figscale, 6 * figscale))
        for i, gene in enumerate(gene_lookup['gene_name']):
            ax = axes.flat[i]
            subtrace = nodrug_itrace.sel(day=day, gene=gene)
            bf_dict, out_ax = az.plot_bf(subtrace, 'diff', ax=ax)
            nodrug_bayes_factors[day][gene] = bf_dict
            ax.set_xlabel("diff {} day {}".format(gene, day))
        pyplot.savefig("day_{}_survival_bayes_factors.svg", bbox_inches="tight")
        pyplot.show()
    
    nodrug_genesum = az.summary(nodrug_itrace, ['gene_level'], coords={'gene': 'siCtrl'})
    control_hdibounds = list(nodrug_genesum.loc['gene_level[placeholder]', 
                                           ['hdi_3%', 'hdi_97%']])
    control_mean = nodrug_genesum.loc['gene_level[placeholder]', 'mean']
    az.plot_posterior(nodrug_itrace,
                      ['gene_level'],
                      coords={'placeholder': 'placeholder'},
                      rope=control_hdibounds,
                      ref_val=control_mean,
                      grid=(6,7))
    pyplot.savefig("gene_level_posteriors.svg", bbox_inches="tight")
    pyplot.show()
    
    fis_idx = (treat_lookup
               .filter(
                   pl.col('treatment').eq('Fisetin')
                   )
               .select(pl.col('treat_idnum'))
               )[0,0]
    quer_idx = (treat_lookup
               .filter(
                   pl.col('treatment').eq('Quercetin')
                   )
               .select(pl.col('treat_idnum'))
               )[0,0]
    
    # not using day 0 data because no drug treatment conditions measured 
    # on day 0
    treat_table = (table2
                   .filter(
                       pl.col('time').gt(0) &
                       pl.col('treatment').eq('untreated').not_()
                       )
                   )
    coords2 = dict(coords)
    coords2['day'] = list(filter(lambda x : x > 0, coords['day']))
    coords2['treatment'] = list(filter(lambda x : x != 'untreated', coords['treatment']))
    treat_lookup2 = treat_lookup.filter(pl.col('treatment').eq('untreated').not_())
    
    with pm.Model(coords=coords2) as drug_model:
        std_lum = pm.Data('std_lum', treat_table['std_lum'])
        # need to convert from times to offsets
        time = pm.Data('time', treat_table['time'] - 1)
        gene_idx = pm.Data('gene_idx', treat_table['gene_idnum'])
        treat_idx = pm.Data('treat_idx', treat_table['treat_idnum'])

        post_baseline = pm.Normal('post_baseline', mu=0, sigma=1)
        drug_base = pm.Normal('drug_base', 
                              mu=post_baseline, 
                              sigma=1,
                              shape=len(treat_lookup2),
                              dims=['treatment'])
        interact = pm.Normal('interact',
                             mu=drug_base[:, np.newaxis],
                             sigma=1,
                             shape=(len(treat_lookup2), len(gene_lookup)),
                             dims=['treatment', 'gene'])
        interact_day = pm.Normal('interact_day',
                                 mu=interact[..., np.newaxis],
                                 sigma=1,
                                 shape=(len(treat_lookup2), len(gene_lookup), 3),
                                 dims=['treatment', 'gene', 'day'])
        
        resp = interact_day[treat_idx, gene_idx, time]
        tau = pm.Gamma('tau', alpha=100, beta=0.1)
        level = pm.Normal('level',
                          mu=resp,
                          tau=tau,
                          observed=std_lum)
        
        fis_line = pm.Deterministic('fis_line',
                                    interact_day[fis_idx, sictrl_idx],
                                    dims=['day'])
        quer_line = pm.Deterministic('quer_line',
                                    interact_day[quer_idx, sictrl_idx],
                                    dims=['day'])
        
        fis_diff = pm.Deterministic('fis_diff',
                                  interact_day[fis_idx] - fis_line[np.newaxis, :],
                                  dims=['gene', 'day'])
        
        quer_diff = pm.Deterministic('quer_diff',
                                     interact_day[quer_idx] - quer_line[np.newaxis, :],
                                     dims=['gene', 'day'])
    
    
    with drug_model:
        prior = pm.sample_prior_predictive()
        drug_itrace = pm.sample(draws=6000,
                                  nuts_sampler='blackjax',
                                  nuts={'target_accept': 0.95})
        post = pm.sample_posterior_predictive(drug_itrace)
        like = pm.compute_log_likelihood(drug_itrace)
    drug_itrace.extend(prior)
    drug_itrace.extend(post)
    
    drug_graph = drug_model.to_graphviz()
    drug_graph.render('drug_model.gv', format='svg')
    
    az.plot_energy(drug_itrace)
    pyplot.title("Drug model energy diagnostic")
    pyplot.savefig("drug_energy_diag.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_loo_pit(drug_itrace, 'level')
    pyplot.title("LOO PIT plot for drug model")
    pyplot.savefig("loo_pit_drug.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_ppc(drug_itrace)
    pyplot.title("Drug PPC plot")
    pyplot.savefig("drug_ppc.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_bpv(drug_itrace, 'p_value')
    pyplot.title("Drug Bayesian P Value")
    pyplot.savefig("drug_bayesian_p_val.svg", bbox_inches="tight")
    pyplot.show()
    
    # treated conditions aren't measured on day 0
    
    az.plot_posterior(drug_itrace, 
                      ['fis_diff'], 
                      coords={'day': 1, 'gene': tested_genes},
                      grid=(6,7),
                      ref_val=0)
    pyplot.show()
    az.plot_posterior(drug_itrace, 
                      ['fis_diff'], 
                      coords={'day': 2, 'gene': tested_genes},
                      grid=(6,7),
                      ref_val=0)
    pyplot.show()
    az.plot_posterior(drug_itrace, 
                      ['fis_diff'], 
                      coords={'day': 3, 'gene': tested_genes},
                      grid=(6,7),
                      ref_val=0)
    pyplot.show()
    
    drug_bayes_factors = {}
    drug_bayes_factors['Fisetin'] = {}
    for day in coords2['day']:
        drug_bayes_factors['Fisetin'][day] = {}
        fig, axes = pyplot.subplots(6, 7, figsize=(7 * figscale, 6 * figscale))
        for i, gene in enumerate(gene_lookup['gene_name']):
            ax = axes.flat[i]
            subtrace = drug_itrace.sel(treatment='Fisetin',
                                       day=day,
                                       gene=gene)
            bf, out_ax = az.plot_bf(subtrace, ['fis_diff'], ax=ax)
            drug_bayes_factors['Fisetin'][day][gene] = bf
            ax.set_xlabel("fis diff {} day {}".format(gene, day))
        pyplot.savefig("fisetin_diffs_day_{}_posterior.svg".format(day), bbox_inches="tight")
        pyplot.show()
    
    drug_bayes_factors['Quercetin'] = {}
    for day in coords2['day']:
        drug_bayes_factors['Quercetin'][day] = {}
        fig, axes = pyplot.subplots(6, 7, figsize=(7 * figscale, 6 * figscale))
        for i, gene in enumerate(gene_lookup['gene_name']):
            ax = axes.flat[i]
            subtrace = drug_itrace.sel(treatment='Quercetin',
                                       day=day,
                                       gene=gene)
            bf, out_ax = az.plot_bf(subtrace, ['quer_diff'], ax=ax)
            drug_bayes_factors['Quercetin'][day][gene] = bf
            ax.set_xlabel("quer diff {} day {}".format(gene, day))
        pyplot.savefig("quer_diffs_day_{}_posterior.svg".format(day), bbox_inches="tight")
        pyplot.show()
    
    drug_summary_table = az.summary(drug_itrace,
                                    ['fis_diff', 'quer_diff'],
                                    coords={'gene': tested_genes})
    
    name_matcher = re.compile(r"\[(.+),")
    day_matcher = re.compile(r",\W*(\d+)]")
    
    bf10_col = []
    bf01_col = []
    treat_col = []
    gene_col = []
    day_col = []
    
    for idx_loc in drug_summary_table.index:
        if 'quer' in idx_loc:
            treat_id = 'Quercetin'
        if 'fis' in idx_loc:
            treat_id = 'Fisetin'
        name_match = name_matcher.search(idx_loc)
        gene_name = name_match.group(1)
        day_match = day_matcher.search(idx_loc)
        day_str_match = day_match.group(1)
        day_num = int(day_str_match)
        bf10 = drug_bayes_factors[treat_id][day_num][gene_name]['BF10']
        bf01 = drug_bayes_factors[treat_id][day_num][gene_name]['BF01']
        bf10_col.append(bf10)
        bf01_col.append(bf01)
        treat_col.append(treat_id)
        gene_col.append(gene_name)
        day_col.append(day_num)
    
    drug_summary_table['BF_10'] = bf10_col
    drug_summary_table['BF_01'] = bf01_col
    drug_summary_table['test_drug'] = treat_col
    drug_summary_table['gene'] = gene_col
    drug_summary_table['day'] = day_col
    
    drug_summary_table.to_csv("treated_silencing_diff_from_treat_only_posterior.csv")
    
    evid_silen_impair = drug_summary_table[(drug_summary_table['mean'] > 0) &
                                           (drug_summary_table['BF_10'] > 10)]
    silen_impair_count = evid_silen_impair.value_counts(subset=['test_drug',
                                                                'gene']).reset_index()
    silen_impair_hits = silen_impair_count[silen_impair_count['count'] >= 2].sort_values(['test_drug', 'gene'])
    
    fisetin_impair_bayesian = set()
    quercetin_impair_bayesian = set()
    
    for i in range(len(silen_impair_hits)):
        row = silen_impair_hits.iloc[i]
        if row['test_drug'] == 'Fisetin':
            fisetin_impair_bayesian.add(row['gene'])
        if row['test_drug'] == 'Quercetin':
            quercetin_impair_bayesian.add(row['gene'])
    
    fig, ax = pyplot.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    bayes_venn = venn.venn3([bayesian_survival_implicated,
                fisetin_impair_bayesian,
                quercetin_impair_bayesian],
               set_labels=['Survival', 'Fisetin', 'Quercetin'],
               ax=ax)
    bayes_venn.get_label_by_id('A').set_fontweight('bold')
    bayes_venn.get_label_by_id('B').set_fontweight('bold')
    bayes_venn.get_label_by_id('C').set_fontweight('bold')
    
    surv_only_label = '\n'.join(
        map(lambda pair : '; '.join(pair),
            itertools.batched(filter(lambda name : name not in ['FMNL3', 'PPP1R12C'],
                                     bayesian_survival_implicated - (fisetin_impair_bayesian | quercetin_impair_bayesian)
                                     ),
                              2)
            )
        ) + '\n\nControls:\n{}'.format(['FMNL3', 'PPP1R12C'])
    
    bayes_venn.get_label_by_id('100').set_text(surv_only_label)
    
    fis_only_label = '\n'.join(
        map(lambda pair: '; '.join(pair),
            itertools.batched(fisetin_impair_bayesian - (bayesian_survival_implicated | quercetin_impair_bayesian),
                              2
                              )
            )
        )
    
    quer_and_fis_label = '\n'.join(
        map(lambda pair: '; '.join(pair),
            itertools.batched((fisetin_impair_bayesian & quercetin_impair_bayesian) - bayesian_survival_implicated,
                              2
                              )
            )
        )
    
    bayes_venn.get_label_by_id('010').set_text(fis_only_label)
    bayes_venn.get_label_by_id('011').set_text(quer_and_fis_label)
    
    pyplot.title("Semi-Bayesian Identifications", fontsize=20)
    pyplot.savefig("Semi_Bayesian_identifications.svg", bbox_inches="tight")
    pyplot.show()
    
    fisetin_exceedance = (drug_itrace.posterior['fis_diff'] > 0).mean(('chain', 'draw', 'day'))
    quercetin_exceedane = (drug_itrace.posterior['quer_diff'] > 0).mean(('chain', 'draw', 'day'))
    
    fis_ax = sb.barplot(x=fisetin_exceedance.gene, y=fisetin_exceedance)
    pyplot.xticks(rotation=90)
    pyplot.axhline(0.65, color='purple')
    pyplot.ylabel("Probability silencing impairs drug action")
    pyplot.title("Gene silencing impairs Fisetin senolysis - Bayesian")
    pyplot.show()
    