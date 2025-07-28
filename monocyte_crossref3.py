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
    #table = table.filter(pl.col('gene_name').eq('CCDC174').or_(pl.col('gene_name').eq('ATP5J2')).not_())
    table = table.with_columns(gene_idnum=pl.col('gene_name').rank('dense').sub(1),
                               treat_idnum=pl.col('treatment').rank('dense').sub(1))
    return table


def display_dunnett(names, pvalues):
    ax = sb.barplot(x=names, y=pvalues)
    ax.axhline(y=0.05, color='red')
    pyplot.xticks(rotation=90)


def frequentist_analysis(table):
    # gene comparisons are easy
    gene_only = table.filter(pl.col('time').eq(0)).group_by(['gene_name']).agg(pl.col('log_lum'))
    gene_ctrl = gene_only.filter(pl.col('gene_name').eq('siCtrl'))
    gene_test = gene_only.filter(pl.col('gene_name').eq('siCtrl').not_())
    gene_anova = scipy.stats.f_oneway(*gene_only['log_lum'])
    gene_dunnett = scipy.stats.dunnett(*gene_test['log_lum'], 
                                       control=gene_ctrl['log_lum'][0], 
                                       alternative='less')
    print("A gene differs:", gene_anova)
    print("These genes differ")
    gtested = gene_test.with_columns(dunnett_p = gene_dunnett.pvalue)
    print(gtested.filter(pl.col('dunnett_p').lt(0.01)).select(pl.col('gene_name'), pl.col('dunnett_p')))
    display_dunnett(gene_test['gene_name'], gene_dunnett.pvalue)
    pyplot.axhline(0.05, color='red')
    pyplot.title("Dunnet Test genes")
    pyplot.savefig("posthoc_gene_diff.svg", bbox_inches="tight")
    pyplot.show()
    
    gene_table = pl.from_dict({'gene': gene_test['gene_name'],
                               'dunnett_pvalue': gene_dunnett.pvalue})

    days = []
    fis_anova_p = []
    quer_anova_p = []
    dmso_anova_p = []
    
    dunnett_rows = []
    
    for day in range(1, 4):
        day_i = table.filter(pl.col('time').eq(day))
        fisetin = day_i.filter(pl.col('treatment').eq('Fisetin'))
        quercetin = day_i.filter(pl.col('treatment').eq('Quercetin'))
        dmso = day_i.filter(pl.col('treatment').eq('DMSO'))
        fis = fisetin.group_by(['gene_name']).agg(pl.col('log_lum')).sort(pl.col('gene_name'))
        quer = quercetin.group_by(['gene_name']).agg(pl.col('log_lum')).sort(pl.col('gene_name'))
        dms = dmso.group_by(['gene_name']).agg(pl.col('log_lum')).sort(pl.col('gene_name'))
        
        base_control = dms.filter(pl.col('gene_name').eq('siCtrl'))
        
        dmso_anova = scipy.stats.f_oneway(*dms['log_lum'])
        fis_anova = scipy.stats.f_oneway(*fis['log_lum'])
        quer_anova = scipy.stats.f_oneway(*quer['log_lum'])
        
        days.append(day)
        fis_anova_p.append(fis_anova.pvalue)
        quer_anova_p.append(quer_anova.pvalue)
        dmso_anova_p.append(dmso_anova.pvalue)
        
        print(f'difference in fisetin on day {day}:', fis_anova)
        print(f'difference in quercetin on day {day}:', quer_anova)
        
        fis_test = fis.filter(pl.col('gene_name').eq('siCtrl').not_())
        fis_ctrl = fis.filter(pl.col('gene_name').eq('siCtrl'))
        fis_dunnett = scipy.stats.dunnett(*fis_test['log_lum'],
                                          control=fis_ctrl['log_lum'][0])
        display_dunnett(fis_test['gene_name'], fis_dunnett.pvalue)
        pyplot.title(f"Day {day} Fisetin pairwise")
        pyplot.savefig(f"posthoc_fisetin_day_{day}_pairwise.svg", bbox_inches="tight")
        pyplot.show()
        
        quer_test = quer.filter(pl.col('gene_name').eq('siCtrl').not_())
        quer_ctrl = quer.filter(pl.col('gene_name').eq('siCtrl'))
        quer_dunnett = scipy.stats.dunnett(*quer_test['log_lum'],
                                           control=quer_ctrl['log_lum'][0])
        display_dunnett(quer_test['gene_name'], quer_dunnett.pvalue)
        pyplot.title(f"Day {day} Quercetin pairwise")
        pyplot.savefig(f"posthoc_quercetin_day_{day}_pairwise.svg", bbox_inches="tight")
        pyplot.show()
        
        for i,name in enumerate(fis_test['gene_name']):
            pval = fis_dunnett.pvalue[i]
            row = {'day': day, 'treatment': 'Fisetin', 'gene': name, 'dunnett_pvalue': pval}
            dunnett_rows.append(row)
        for i, name in enumerate(quer_test['gene_name']):
            pval = quer_dunnett.pvalue[i]
            row = {'day': day, 'treatment': 'Quercetin', 'gene': name, 'dunnett_pvalue': pval}
            dunnett_rows.append(row)
        
        fis_dunnett2 = scipy.stats.dunnett(*fis_test['log_lum'],
                                           control=base_control['log_lum'][0])
        quer_dunnett2 = scipy.stats.dunnett(*quer_test['log_lum'],
                                            control=base_control['log_lum'][0])
        display_dunnett(fis_test['gene_name'], fis_dunnett2.pvalue)
        pyplot.axhline(y=0.20)
        pyplot.title(f"Day {day} Fisetin silencing versus control-control")
        pyplot.savefig(f"posthoc_fisetin_vs_controlcontrol_day_{day}.svg", bbox_inches="tight")
        pyplot.show()
        display_dunnett(quer_test['gene_name'], quer_dunnett2.pvalue)
        pyplot.axhline(y=0.20)
        pyplot.title(f"Day {day} Quercetin silencing versus control-control")
        pyplot.savefig(f"posthoc_quercetin_vs_controlcontrol_day_{day}.svg", bbox_inches="tight")
        pyplot.show()
        
        
    #breakpoint()
    dunnett_table = pl.from_dicts(dunnett_rows)
    
    anova_table = pl.from_dict({'day': days,
                                'fis_anova_pvalue': fis_anova_p,
                                'quer_anova_pvalue': quer_anova_p,
                                'dmso_anova_pvalue': dmso_anova_p})
    return dunnett_table, anova_table, gene_table


if __name__ == '__main__':
    
    table = load_data()
    
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
    
    basetab = table.filter(
        pl.col('treatment').eq('DMSO').and_(pl.col('gene_name').eq('siCtrl'))
        ).select(pl.col('time'),base_lum=pl.col('luminescence').mean().over(pl.col('time'))).unique()
    table = table.join(basetab, on=pl.col('time'))
    
    meanlog = table['luminescence'].log().mean()
    stdlog = table['luminescence'].log().std()
    
    table = table.with_columns(
        scale_lum_delta = (pl.col('luminescence').log() - meanlog) / stdlog
        )
    
    sb.scatterplot(table.filter(pl.col('treatment').eq('DMSO').and_(pl.col('gene_name').eq('siCtrl'))),
                   x='time',
                   y='luminescence')
    pyplot.title("siCtrl + DMSO luminescence")
    pyplot.savefig('control_luminescence.svg')
    pyplot.show()
    
    table = table.with_columns(
        mm_lum=(pl.col('luminescence') - pl.col('luminescence').min())/
        (pl.col('luminescence').max() - pl.col('luminescence').min()),
        log_lum = pl.col('luminescence').log(),
        log_time = pl.col('time').log1p(),
        std_lum=(pl.col('luminescence') - pl.col('luminescence').mean())/pl.col('luminescence').std()
        )
    
    control_tab = table.filter(
        pl.col('treatment').eq('DMSO').and_(
            pl.col('gene_name').eq('siCtrl'))
        ).select(pl.col('time'),base_lum=pl.col('luminescence').mean().over(pl.col('time'))).unique()
    
    table = table.join(control_tab, on=pl.col('time'))
    table = table.with_columns(fold_lum=pl.col('luminescence')/pl.col('base_lum'))
    

    
    _dunnett_table, _anova_table, _gene_table = frequentist_analysis(table)
    
    
    
    with pyplot.rc_context(rc={'legend.fontsize': 30}):
        relplot = sb.relplot(table,
                             x='time',
                             y='luminescence',
                             hue='treatment',
                             col='gene_name',
                             col_wrap=6,
                             kind='line')
        for ax in relplot.figure.axes:
            ax.title.set_fontsize(30)
        pyplot.savefig('luminance_vs_time.svg')
        pyplot.show()
        
        relplot2 = sb.relplot(table,
                              x='time',
                              y='fold_lum',
                              hue='treatment',
                              col='gene_name',
                              col_wrap=6,
                              kind='line')
        for ax in relplot2.figure.axes:
            ax.title.set_fontsize(30)
        pyplot.savefig('foldlum_over_time.svg')
        pyplot.show()
    
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
    
    
    with pm.Model(coords=coords) as model:
        fold_luminescence = pm.Data('fold_luminescence', table['fold_lum'])
        time = pm.Data('time', table['time'])
        gene_idx = pm.Data('gene_idx', table['gene_idnum'])
        treat_idx = pm.Data('treat_idx', table['treat_idnum'])
        
        base_const = pm.Normal('base_const',
                               mu=1,
                               sigma=1)
        gene_const = pm.Normal('gene_const',
                               mu=base_const,
                               sigma=1,
                               #lower=0,
                               shape=len(gene_lookup),
                               dims=['gene'])
        drug_step = pm.Normal('drug_step',
                              mu=0,
                              sigma=1,
                              shape=(len(treat_lookup), 1), 
                              dims=['treatment', 'placeholder'])
        interact_step = pm.Normal('interact_step',
                                  mu=drug_step,
                                  sigma=1,
                                  shape=(len(treat_lookup), len(gene_lookup)),
                                  dims=['treatment', 'gene'])
        
        drug_slope = pm.Normal('drug_slope', 
                               mu=0, 
                               sigma=1, 
                               shape=(len(treat_lookup), 1), 
                               dims=['treatment', 'placeholder'])
        interact_slope = pm.Normal('interact_slope',
                                   mu=drug_slope,
                                   sigma=1,
                                   shape=(len(treat_lookup), len(gene_lookup)),
                                   dims=['treatment', 'gene'])
        
        gconst = gene_const[gene_idx]
        istep = pm.math.where(time > 0, interact_step[treat_idx, gene_idx], 0)
        lin = interact_slope[treat_idx, gene_idx] * time
        lin_eff = pm.math.where(time > 0, lin, 0)
        resp = gconst + istep + lin_eff
        
        sigma = pm.Exponential('sigma', lam=1)
        
        likelihood = pm.Normal('likelihood',
                               mu=resp,
                               sigma=sigma,
                               observed=fold_luminescence)
        
    
    with model:
        prior = pm.sample_prior_predictive(draws=2000)
        itrace = pm.sample(draws=5000,
                           nuts_sampler='blackjax',
                           nuts={'target_accept': 0.95})
        post = pm.sample_posterior_predictive(itrace)
        loglike = pm.compute_log_likelihood(itrace)
    itrace.extend(prior)
    itrace.extend(post)
    
    y_pred = itrace.posterior_predictive.stack(sample=('chain', 'draw'))['likelihood'].values.T
    y_act = itrace.constant_data['fold_luminescence'].values
    r2_tab = az.r2_score(y_act, y_pred)
    print("\n", "In sample explained variance")
    print("R2", r2_tab.r2)
    print("R2 stddev", r2_tab.r2_std)
        
    modelgraph = model.to_graphviz()
    modelgraph.render("modelgraph.gv", format="svg")
    
    az.plot_energy(itrace)
    pyplot.title("Energy Diagnostic Plot")
    pyplot.savefig("energy_diag.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_ppc(itrace)
    pyplot.title("Posterior Predictive Check")
    pyplot.savefig("ppc.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_loo_pit(itrace, 'likelihood', n_unif=500)
    pyplot.title("LOO-PIT plot")
    pyplot.savefig("loo_pit.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_bpv(itrace, kind="p_value")
    pyplot.title("Bayesian P-value")
    pyplot.savefig("bayesian_pval.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_trace(itrace)
    pyplot.savefig("TracePlot.png", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace, ['sigma'])
    pyplot.savefig("stddev_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_bf(itrace, 'base_const', ref_val=1)
    pyplot.savefig("base_const_bf.svg", bbox_inches="tight")
    pyplot.show()
    
    
    for gene_name in gene_lookup['gene_name']:
        g_trace = itrace.sel(gene=gene_name)
        az.plot_bf(g_trace, 'gene_const', ref_val=1)
        pyplot.xlabel(f"gene_const {gene_name}")
        pyplot.savefig(f"gene_const/gene_const_{gene_name}.svg", bbox_inches="tight")
        pyplot.show()
    
    control_const_tab = az.summary(itrace, ['gene_const'], coords={'gene': 'siCtrl'})
    control_const_lowbound = control_const_tab.loc['gene_const', 'hdi_3%']
    az.plot_posterior(itrace, ['gene_const'], ref_val=control_const_lowbound, grid=(6,7))
    pyplot.savefig("gene_const/gene_const_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_step'],
                      ref_val=0,
                      coords={'treatment': 'DMSO'},
                      grid=(6,7))
    pyplot.savefig("interact_step_DMSO_absence.svg", bbox_inches="tight")
    pyplot.show()
    
    for gene_name in gene_lookup['gene_name']:
        for drug_name in treat_lookup['treatment']:
            sub_trace = itrace.sel(gene=gene_name, treatment=drug_name)
            az.plot_bf(sub_trace, 'interact_step', ref_val=0)
            pyplot.xlabel(f"interact_step {gene_name} {drug_name}")
            pyplot.savefig(f"step_bf/interact_step_{gene_name}_{drug_name}_bf.svg", bbox_inches="tight")
            pyplot.show()
    
    control_step_tab = az.summary(itrace, ['interact_step'], coords={'gene': 'siCtrl'})
    dmso_step_highbound = control_step_tab.loc['interact_step[DMSO]', 'hdi_97%']
    az.plot_posterior(itrace, 
                      ['interact_step'], 
                      ref_val=dmso_step_highbound, 
                      coords={'treatment': 'DMSO'},
                      grid=(6,7))
    pyplot.savefig("interact_step_DMSO_posterior.svg", bbox_inches="tight")
    pyplot.show()
    fisetin_step_highbound = control_step_tab.loc['interact_step[Fisetin]', 'hdi_97%']
    az.plot_posterior(itrace,
                      ['interact_step'],
                      ref_val=fisetin_step_highbound,
                      coords={'treatment': 'Fisetin'},
                      grid=(6,7))
    pyplot.savefig("interact_step_Fisetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    quercetin_step_highbound = control_step_tab.loc['interact_step[Quercetin]', 'hdi_97%']
    az.plot_posterior(itrace,
                      ['interact_step'],
                      ref_val=quercetin_step_highbound,
                      coords={'treatment': 'Quercetin'},
                      grid=(6,7))
    pyplot.savefig("interact_step_Quercetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    for gene_name in gene_lookup['gene_name']:
        for drug_name in treat_lookup['treatment']:
            sub_trace = itrace.sel(gene=gene_name, treatment=drug_name)
            az.plot_bf(sub_trace, 'interact_slope', ref_val=0)
            pyplot.xlabel(f"interact_slope {gene_name} {drug_name}")
            pyplot.savefig(f"interact_slope_bf/interact_slope_{gene_name}_{drug_name}_bf.svg",
                           bbox_inches="tight")
            pyplot.show()
    
    control_slope_tab = az.summary(itrace, ['interact_slope'], coords={'gene': 'siCtrl'})
    fis_ctrl_slope_bounds = control_slope_tab.loc['interact_slope[Fisetin]', ['hdi_3%', 'hdi_97%']]
    quer_ctrl_slope_bounds = control_slope_tab.loc['interact_slope[Quercetin]', ['hdi_3%', 'hdi_97%']]
    dmso_ctrl_slope_bounds = control_slope_tab.loc['interact_slope[DMSO]', ['hdi_3%', 'hdi_97%']]
    
    az.plot_posterior(itrace, 
                      ['interact_slope'],
                      ref_val=0,
                      coords={'treatment': 'DMSO'},
                      grid=(6,7),
                      rope=dmso_ctrl_slope_bounds)
    pyplot.savefig("interact_slope_DMSO_posterior.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_posterior(itrace,
                      ['interact_slope'],
                      ref_val=0,
                      coords={'treatment': 'Fisetin'},
                      grid=(6,7),
                      rope=fis_ctrl_slope_bounds)
    pyplot.savefig("interact_slope_Fisetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    az.plot_posterior(itrace,
                      ['interact_slope'],
                      ref_val=0,
                      coords={'treatment': 'Quercetin'},
                      grid=(6,7),
                      rope=quer_ctrl_slope_bounds)
    pyplot.savefig("interact_slope_Quercetin_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    

