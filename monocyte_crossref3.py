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
    dunnett_table = []
    for day in range(1, 4):
        daytab = table.filter(pl.col('time').eq(day))
        for gene in daytab['gene_name'].unique():
            geneday = daytab.filter(pl.col('gene_name').eq(gene))
            treatgroups = geneday.group_by(['treatment']).agg(pl.col('log_lum')).sort(pl.col('treatment'))
            geneday_anova = scipy.stats.f_oneway(*treatgroups['log_lum'])
            anova_row = {'day': day, 
                         'gene': gene, 
                         'anova_stat': geneday_anova.statistic,
                         'anova_pvalue': geneday_anova.pvalue}
            anova_table.append(anova_row)
            if geneday_anova.pvalue <= 0.05:
                dmso = treatgroups.filter(pl.col('treatment').eq('DMSO'))['log_lum'][0]
                treatments = treatgroups.filter(pl.col('treatment').eq('DMSO').not_())
                geneday_dunnett = scipy.stats.dunnett(*treatments['log_lum'], control=dmso)
                dunnett_row0 = {'day': day,
                               'gene': gene,
                               'treatments': treatments['treatment'][0],
                               'dunnett_stat': geneday_dunnett.statistic[0],
                               'dunnett_pvalue': geneday_dunnett.pvalue[0]}
                dunnett_row1 = {'day': day,
                               'gene': gene,
                               'treatments': treatments['treatment'][1],
                               'dunnett_stat': geneday_dunnett.statistic[1],
                               'dunnett_pvalue': geneday_dunnett.pvalue[1]}
                dunnett_table.append(dunnett_row0)
                dunnett_table.append(dunnett_row1)
    anova_ptable = pl.from_dicts(anova_table)
    dunnett_ptable = pl.from_dicts(dunnett_table)
    return anova_ptable, dunnett_ptable


def frequentist_genediff(table):
    # gene comparisons are easyt
    gene_only = table.filter(pl.col('time').eq(0)).group_by(['gene_name']).agg(pl.col('log_lum'))
    gene_ctrl = gene_only.filter(pl.col('gene_name').eq('siCtrl'))
    gene_test = gene_only.filter(pl.col('gene_name').eq('siCtrl').not_())
    gene_anova = scipy.stats.f_oneway(*gene_only['log_lum'])
    gene_dunnett = scipy.stats.dunnett(*gene_test['log_lum'], 
                                       control=gene_ctrl['log_lum'][0], 
                                       alternative='less')
    gene_table = pl.from_dict({'gene': gene_test['gene_name'],
                               'dunnett_pvalue': gene_dunnett.pvalue,
                               'dunnett_stat': gene_dunnett.statistic})
    return gene_table, gene_anova


def frequentist_analysis(table):
    gene_table, gene_anova = frequentist_genediff(table)
    print(gene_anova)
    sb.barplot(gene_table, x='dunnett_pvalue', y='gene')
    pyplot.xticks(rotation=90)
    pyplot.axvline(0.05, color='red')
    pyplot.savefig("gene_diff_frequentist.svg", bbox_inches="tight")
    pyplot.show()
    anova2_table, dunnett2_table = frequentist_treatment_impair(table)
    grid = sb.catplot(anova2_table,
               x='anova_pvalue',
               y='gene',
               col='day',
               kind='bar')
    shape = grid.axes.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax = grid.axes[i,j]
            ax.axvline(0.20, color='blue')
            ax.axvline(0.05, color='red')
    pyplot.savefig("gene_treat_diff_detectable.svg", bbox_inches="tight")
    pyplot.show()
    
    grid = sb.catplot(dunnett2_table,
                      x='dunnett_pvalue',
                      y='gene',
                      col='day',
                      row='treatments',
                      kind='bar')
    for ax in grid.axes.flat:
        ax.axvline(0.05, color='red')
    pyplot.savefig("post_hoc_diffs.svg", bbox_inches="tight")
    pyplot.show()



if __name__ == '__main__':
    
    table = load_data()
    #breakpoint()
    table = table.with_columns(time2=pl.col('time').pow(2))
    gene_lookup = table.select(pl.col('gene_idnum'),pl.col('gene_name')).unique()
    gene_lookup = gene_lookup.sort(pl.col('gene_idnum'))
    treat_lookup = table.select(pl.col('treat_idnum'), pl.col('treatment')).unique()
    treat_lookup = treat_lookup.sort(pl.col('treat_idnum'))
    coords = {'gene': gene_lookup['gene_name'], 
              'treatment': treat_lookup['treatment'],
              'placeholder': ['placeholder']}
    
    dmso_idx = treat_lookup.filter(
        pl.col('treatment').eq('DMSO')
        ).select(pl.col('treat_idnum'))[0,0]
    fisetin_idx = treat_lookup.filter(
        pl.col('treatment').eq('Fisetin')
        ).select(pl.col('treat_idnum'))[0,0]
    quercetin_idx = treat_lookup.filter(
        pl.col('treatment').eq('Quercetin')
        ).select(pl.col('treat_idnum'))[0,0]
    sictrl_idx = gene_lookup.filter(
        pl.col('gene_name').eq('siCtrl')
        ).select(pl.col('gene_idnum'))[0,0]
    
    
    meanlog = table['luminescence'].log().mean()
    stdlog = table['luminescence'].log().std()
    
    table = table.with_columns(
        scale_lum_delta = (pl.col('luminescence').log() - meanlog) / stdlog
        )
    
    sb.regplot(table.filter(pl.col('treatment').eq('DMSO').and_(pl.col('gene_name').eq('siCtrl'))),
                   x='time',
                   y='luminescence')
    pyplot.title("siCtrl + DMSO luminescence")
    pyplot.savefig('control_luminescence.svg')
    pyplot.show()
    
    
    table = table.with_columns(
        mm_lum=(pl.col('luminescence') - pl.col('luminescence').min())/
        (pl.col('luminescence').max() - pl.col('luminescence').min()),
        log_lum = pl.col('luminescence').log(base=2),
        log_time = pl.col('time').log1p() / np.log(2),
        std_lum=(pl.col('luminescence') - pl.col('luminescence').mean())/pl.col('luminescence').std(),
        )
    
    frequentist_analysis(table)
    
    table0 = table.filter(pl.col('time').eq(0))
    control0 = table0.filter(pl.col('treatment').eq('untreated').and_(pl.col('gene_name').eq('siCtrl')))
    control0mean = control0.select(pl.col('luminescence').mean())[0,0]
    control0std = control0.select(pl.col('luminescence').std())[0,0]
    table0 = table0.with_columns(cstd_lum = (pl.col('luminescence') - control0mean) / control0std)
    
    with pm.Model(coords=coords) as gene_only_model:
        std_luminescence = pm.Data('cstd_luminescence', table0['cstd_lum'])
        gene_idx = pm.Data('gene_idx', table0['gene_idnum'])
        base_const = pm.Normal('base_const',
                               mu=0,
                               sigma=1)
        gene_const = pm.Normal('gene_const',
                               mu=base_const,
                               sigma=1,
                               shape=len(gene_lookup),
                               dims=['gene'])
        gconst = gene_const[gene_idx]
        
        tau = pm.Gamma('tau', alpha=50, beta=0.01)
        
        like = pm.Normal('like',
                         mu=gconst,
                         tau=tau,
                         observed=std_luminescence)
    
    with gene_only_model:
        gene_prior = pm.sample_prior_predictive()
        gene_itrace = pm.sample(draws=6000,
                           nuts_sampler='blackjax',
                           nuts={'target_accept': 0.95})
        gene_pp = pm.sample_posterior_predictive(gene_itrace)
        gene_loglike = pm.compute_log_likelihood(gene_itrace)
    gene_itrace.extend(gene_prior)
    gene_itrace.extend(gene_pp)
    
    az.plot_energy(gene_itrace)
    pyplot.title("Gene Viability Energy Diagnostic")
    pyplot.savefig("gene_viability_energy_diagnostic.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_ppc(gene_itrace)
    pyplot.title("Gene Viability PPC Plot")
    pyplot.xlabel("CSTD Gene Viability (luminescence)")
    pyplot.ylabel("Probability Density")
    pyplot.savefig('gene_viability_ppc_plot.svg', bbox_inches="tight")
    pyplot.show()
    az.plot_loo_pit(gene_itrace, 'like')
    pyplot.title("Gene Viability LOO PIT Plot")
    pyplot.savefig('gene_viability_loo_pit_plot.svg', bbox_inches="tight")
    pyplot.show()
    az.plot_bpv(gene_itrace)
    pyplot.title("Gene Viability Bayesian u Value Plot")
    pyplot.savefig("gene_viability_bayesian_uvalue_plot.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_bpv(gene_itrace, kind='p_value')
    pyplot.title("Gene Viability Bayesian p Value Plot")
    pyplot.xlabel("Overprediction Probability")
    pyplot.savefig('gene_viability_bayesian_pvalue_plot.svg', bbox_inches='tight')
    pyplot.show()
    
    coord_pairs = itertools.product(range(4), range(10))
    space = 10
    fig, axes = pyplot.subplots(4, 10, figsize=(10 *space, 4 * space))
    gene_bfs = {}
    for gene_name in gene_lookup['gene_name']:
        sub_genetrace = gene_itrace.sel(gene = gene_name)
        row, col = next(coord_pairs)
        ax = axes[row, col]
        bfs, ax_out = az.plot_bf(sub_genetrace, 'gene_const', ax=ax)
        gene_bfs[gene_name] = bfs
        ax_out.set_xlabel(f"CSTD gene_const {gene_name}", fontsize=16)
        ax_out.set_title(ax.get_title(), fontsize=20)
        #ax_out.get_legend().remove()
    pyplot.savefig("gene_viability_bayes_factors_plot.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(gene_itrace, 
                      ['gene_const'],
                      ref_val=0,
                      rope=(-2,2),
                      grid=(4,10))
    pyplot.savefig("gene_viability_posterior_comparison.svg", bbox_inches="tight")
    pyplot.show()
    
    gene_only_graph = gene_only_model.to_graphviz()
    gene_only_graph.render("gene_viability_modeling.gv", format="svg")
    
    az.plot_trace(gene_itrace)
    pyplot.savefig("gene_viability_traceplot.png", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(gene_itrace, ['base_const'])
    pyplot.savefig("gene_viability_base_const_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(gene_itrace, ['tau'])
    pyplot.savefig("gene_viability_tau_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    controltab = table.filter(pl.col('treatment').eq('DMSO').or_(pl.col('treatment').eq('untreated'))
                              ).filter(pl.col('gene_name').eq('siCtrl'))
    control_med = controltab.select(pl.col('time'),
                                    medlum=pl.col('luminescence').median().over(pl.col('time'))
                                    ).unique()
    table2 = table.join(control_med, on=pl.col('time')).with_columns(
        fold_lum = pl.col('luminescence') / pl.col('medlum')
        )
    
    
    dmso_idx = treat_lookup.filter(
        pl.col('treatment').eq('DMSO')
        ).select(pl.col('treat_idnum'))[0,0]
    fisetin_idx = treat_lookup.filter(
        pl.col('treatment').eq('Fisetin')
        ).select(pl.col('treat_idnum'))[0,0]
    quercetin_idx = treat_lookup.filter(
        pl.col('treatment').eq('Quercetin')
        ).select(pl.col('treat_idnum'))[0,0]
    sictrl_idx = gene_lookup.filter(
        pl.col('gene_name').eq('siCtrl')
        ).select(pl.col('gene_idnum'))[0,0]
    
    with pm.Model(coords=coords) as step_model:
        std_lum = pm.Data('std_lum', table2['std_lum'])
        gene_idx = pm.Data('gene_idx', table2['gene_idnum'])
        treat_idx = pm.Data('treat_idx', table2['treat_idnum'])
        time = pm.Data('time', table2['time'])
        
        base_const = pm.Normal('base_const', mu=0, sigma=1)
        gene_const = pm.Normal('gene_const',
                               mu=base_const,
                               sigma=1,
                               shape=len(gene_lookup),
                               dims=['gene'])
        treat_step1 = pm.Normal('treat_step1',
                               mu=0,
                               sigma=1,
                               shape=(len(treat_lookup),1),
                               dims=['treatment', 'placeholder'])
        interact_step1 = pm.Normal('interact_step1',
                                   mu=treat_step1,
                                   sigma=1,
                                   shape=(len(treat_lookup), len(gene_lookup)),
                                   dims=['treatment', 'gene'])
        treat_step2 = pm.Normal('treat_step2',
                                mu=0,
                                sigma=1,
                                shape=(len(treat_lookup),1),
                                dims=['treatment', 'placeholder'])
        interact_step2 = pm.Normal('interact_step2',
                                   mu=treat_step2,
                                   sigma=1,
                                   shape=(len(treat_lookup), len(gene_lookup)),
                                   dims=['treatment', 'gene'])
        treat_step3 = pm.Normal('treat_step3',
                                mu=0,
                                sigma=1,
                                shape=len(treat_lookup),
                                dims=['treatment'])
        
        gconst = gene_const[gene_idx]
        treat_resp1 = pm.math.where(time >= 1, interact_step1[treat_idx, gene_idx], 0)
        treat_resp2 = pm.math.where(time >= 2, interact_step2[treat_idx, gene_idx], 0)
        treat_resp3 = pm.math.where(time >= 3, treat_step3[treat_idx], 0)
        resp = gconst + treat_resp1 + treat_resp2 + treat_resp3
        
        tau = pm.Gamma('tau', alpha=50, beta=0.01)
        
        step_like = pm.Normal('step_like',
                              mu = resp,
                              tau=tau,
                              observed = std_lum
                              )
    
    with step_model:
        prior = pm.sample_prior_predictive()
        step_itrace = pm.sample(draws=6000,
                                nuts_sampler='blackjax',
                                nuts={'target_accept': 0.95})
        post = pm.sample_posterior_predictive(step_itrace)
        like = pm.compute_log_likelihood(step_itrace)
    
    step_itrace.extend(prior)
    step_itrace.extend(post)
    
    az.plot_loo_pit(step_itrace, 'step_like')
    pyplot.title("step model LOO PIT")
    pyplot.savefig("step_model_loo_pit.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_ppc(step_itrace)
    pyplot.title("step model PPC")
    pyplot.savefig("step_model_ppc.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_bpv(step_itrace, kind='p_value')
    pyplot.title("step model bayesian P value")
    pyplot.savefig("step_model_bayesian_pvalue.svg", bbox_inches="tight")
    pyplot.show()
    
    day1unsil = az.summary(step_itrace, 'interact_step1', coords={'gene': 'siCtrl'})
    fisbounds = day1unsil.loc['interact_step1[Fisetin]', ['hdi_3%', 'hdi_97%']]
    az.plot_posterior(step_itrace, 
                      ['interact_step1'], 
                      coords={'treatment': 'Fisetin'},
                      rope=fisbounds,
                      ref_val=0,
                      grid=(6,7))
    pyplot.savefig("day1_fisetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    dmsobounds = day1unsil.loc['interact_step1[DMSO]', ['hdi_3%', 'hdi_97%']]
    az.plot_posterior(step_itrace,
                      ['interact_step1'],
                      coords={'treatment': 'DMSO'},
                      rope=dmsobounds,
                      ref_val=0,
                      grid=(6,7))
    pyplot.savefig("day1_dmso_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    querbounds = day1unsil.loc['interact_step1[Quercetin]', ['hdi_3%', 'hdi_97%']]
    az.plot_posterior(step_itrace,
                      ['interact_step1'],
                      coords={'treatment': 'Quercetin'},
                      rope=querbounds,
                      ref_val=0,
                      grid=(6,7))
    pyplot.savefig("day1_quercetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    fis_diffs = step_itrace.sel(treatment='Fisetin').posterior - step_itrace.sel(treatment='DMSO').posterior
    step_itrace.add_groups({'fis_diffs': fis_diffs})
    day1diffsum = az.summary(step_itrace,
                             ['interact_step1'],
                             group='fis_diffs',
                             coords={'gene': 'siCtrl'})
    diffbounds = day1diffsum.loc['interact_step1', ['hdi_3%', 'hdi_97%']]
    
    az.plot_posterior(step_itrace,
                      ['interact_step1'],
                      group='fis_diffs',
                      grid=(6,7),
                      ref_val=0.0,
                      rope=diffbounds)
    pyplot.savefig("day1_fisetin_diff_step_posterior.svg", bbox_inches="tight")
    pyplot.show()
    