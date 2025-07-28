# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:31:54 2025

@author: piercetf
"""

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

    SI_RNA = "db2.xlsx"
    
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
    table = table.filter(pl.col('gene_name').eq('CCDC174').or_(pl.col('gene_name').eq('ATP5J2')).not_())
    table = table.with_columns(gene_idnum=pl.col('gene_name').rank('dense').sub(1),
                               treat_idnum=pl.col('treatment').rank('dense').sub(1))
    return table

    
    

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
                              y='scale_lum_delta',
                              hue='treatment',
                              col='gene_name',
                              col_wrap=6,
                              kind='line')
        for ax in relplot2.figure.axes:
            ax.title.set_fontsize(30)
        pyplot.savefig('scale_lum_delta_vs_time.svg')
        pyplot.show()
    
    with pm.Model(coords=coords) as model:
        time = pm.Data('time', table['time'])
        treat_idx = pm.Data('treat_idx', table['treat_idnum'])
        gene_idx = pm.Data('gene_idx', table['gene_idnum'])
        
        
        base_const = pm.Normal('base_const', mu=0, sigma=1)
        gene_const = pm.Normal('gene_const',
                               mu=base_const,
                               sigma=1,
                               shape=len(gene_lookup),
                               dims=['gene'])
        
        # model the effect of introducing the drugs at time 1 as similar within drugs
        # yet allowed to differ between genes in that structure
        # with our model being that the introduction of the drug produces an immediate
        # die off event, and thus can be modeled using a step function
        drug_off = pm.TruncatedNormal('drug_off',
                              mu=0,
                              sigma=1,
                              upper=0,
                              shape=len(treat_lookup),
                              dims=['treatment'])
        drug_expand = pm.Deterministic('drug_expand',
                                        drug_off[:, np.newaxis],
                                        dims=['treatment', 'placeholder'])
        interact_off = pm.TruncatedNormal('interact_off',
                                  mu=drug_expand,
                                  upper=0,
                                  shape=(len(treat_lookup), len(gene_lookup)),
                                  dims=['treatment', 'gene'])
        interact = pm.math.where(time >= 1, interact_off[treat_idx, gene_idx], 0)
        
        # we think that the drug should also act over time, but the drugs
        # have biological half-lives as they get metabolized and degraded.
        # thus the rate at which the drug can act must decline according to
        # an exponential decay e^(-kt)
        # search of literature reveals that the decay rate parameter we should
        # expect for DMSO ~ 1.39 and for Fisetin, Quercetin ~ 4.75,
        drug_decay_inner = pm.Laplace.dist(
            mu=np.where(treat_lookup['treatment'] == 'DMSO', 1.39, 4.75),
            b=0.1)
        drug_decay = pm.Truncated('drug_decay',
                                  dist=drug_decay_inner,
                                  lower=0,
                                  dims=['treatment'])
        # drug decay rate is assumed to not be influenced by silenced genes.
        decay = drug_decay[treat_idx]
        # exponential decay starts at moment drug is added,
        # so time input has to be shifted to accomodate
        timeshift = time - 1
        # quantity of remaining drug
        dose = pm.math.exp(-decay * timeshift)
        
        # we assume that effect of drug is proportional to dose,
        # but not 1:1 to it. denote A.
        drug_time_effect = pm.Beta('drug_time_effect',
                                   alpha=1,
                                   beta=10,
                                   shape=len(treat_lookup),
                                   dims=['treatment'])
        # we wish to explicitly model that the silenced genes alter the
        # effectiveness of the drugs or not
        drug_time_expand = pm.Deterministic('drug_time_expand',
                                            drug_time_effect[:, np.newaxis],
                                            dims=['treatment', 'placeholder'])
        interact_effect = pm.TruncatedNormal('interact_effect',
                                             mu=drug_time_expand,
                                             sigma=1,
                                             lower=0,
                                             shape=(len(treat_lookup), len(gene_lookup)),
                                             dims=['treatment', 'gene'])
        time_eff = interact_effect[treat_idx, gene_idx]
        # given an exponential decay of dose over time and a fixed
        # ratio between dose and effect, the cumulative effect is 
        # described by (-A/k)exp(-kt) +(A/k), which must be negated
        # to use it as a reduction in viability
        # always positive or zero
        drug_cum_effect = (-time_eff/decay)*dose + (time_eff/decay)
        # always non-positive
        drug_over_time = pm.math.where(time >= 1, -drug_cum_effect, 0)

        # model starvation setting in on day 3, making the assumption that the 
        # effect is approximately the same in linear terms
        starve_effect = pm.Exponential('starve_effect', lam=1)
        starve = pm.math.where(time > 2, starve_effect, 0)
        
        
        response = gene_const[gene_idx] - starve + drug_over_time + interact
        ydata = pm.Data('ydata', table['scale_lum_delta'])
        tau = pm.Gamma('tau', alpha=5, beta=0.1)
        ylik = pm.Normal('ylik', mu=response, tau=tau, observed=ydata)
        
        
        
    
    with model:
        prior = pm.sample_prior_predictive()
        itrace = pm.sample(nuts={'target_accept':0.9}, 
                           nuts_sampler='blackjax',
                           draws=5000)
        post = pm.sample_posterior_predictive(itrace)
        loglike = pm.compute_log_likelihood(itrace)
    itrace.extend(prior)
    itrace.extend(post)
        

    y_pred = itrace.posterior_predictive.stack(sample=('chain', 'draw'))['ylik'].values.T
    y_act = itrace.constant_data['ydata'].values
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
    
    az.plot_loo_pit(itrace, 'ylik', n_unif=500)
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
    
    az.plot_posterior(itrace, ['tau'])
    pyplot.savefig('precision_posterior.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace, ['starve_effect'])
    pyplot.savefig('starvation_posterior.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_bf(itrace, ['base_const'])
    pyplot.savefig('base_const_bf.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace, ['gene_const'], coords={'gene': 'siCtrl'})
    pyplot.title("gene_const siCtrl", fontsize=20)
    pyplot.savefig("gene_const_sictrl_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    g_const_sum = az.summary(itrace, ['gene_const'])
    where_control = g_const_sum.index == 'gene_const[siCtrl]'
    g_const_sictrl = g_const_sum.loc[where_control]
    sictrl_gconst_low = g_const_sictrl.iloc[0]['hdi_3%']
    sictrl_gconst_high = g_const_sictrl.iloc[0]['hdi_97%']
    
    az.plot_posterior(itrace,
                      ['gene_const'],
                      ref_val=sictrl_gconst_low,
                      grid=(6,7))
    pyplot.savefig("below_sictrl_const.svg", bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['gene_const'],
                      ref_val=sictrl_gconst_high,
                      grid=(6,7))
    pyplot.savefig('above_sictrl_const.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['drug_decay'])
    pyplot.savefig("drug_decay_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace, ['drug_off'])
    pyplot.savefig('drug_off_posterior.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace, 
                      ['interact_off'],
                      coords={'treatment': 'DMSO'},
                      grid=(6,7))
    pyplot.savefig('interact_off_dmso_posterior.svg', bbox_inches='tight')
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Fisetin', 'gene': 'siCtrl'})
    pyplot.title("Fisetin siCtrl interact_off", fontsize=20)
    pyplot.savefig("interact_off_fisetin_sictrl_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    interact_off_tab = az.summary(itrace, ['interact_off'])
    where_fisonly = interact_off_tab.index == 'interact_off[Fisetin, siCtrl]'
    interact_fis_only = interact_off_tab.loc[where_fisonly]
    interact_off_fis_lbound = interact_fis_only.iloc[0]['hdi_3%']
    interact_off_fis_hbound = interact_fis_only.iloc[0]['hdi_97%']
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Fisetin'},
                      ref_val=interact_off_fis_hbound,
                      grid=(6,7))
    pyplot.savefig("interact_off_fisetin_above_sictrl_ubound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Fisetin'},
                      ref_val=interact_off_fis_lbound,
                      grid=(6,7))
    pyplot.savefig("interact_off_fisetin_below_sictrl_lbound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['drug_time_effect'])
    pyplot.savefig("drug_time_effect_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Quercetin', 'gene': 'siCtrl'})
    pyplot.title("interact_off\nQuercetin siCtrl", fontsize=20)
    pyplot.savefig("interact_off_quercetin_sictrl_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    where_queronly = interact_off_tab.index == 'interact_off[Quercetin, siCtrl]'
    interact_quer_only = interact_off_tab.loc[where_queronly]
    interact_off_quer_lbound = interact_quer_only.iloc[0]['hdi_3%']
    interact_off_quer_hbound = interact_quer_only.iloc[0]['hdi_97%']
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Quercetin'},
                      ref_val=interact_off_quer_hbound,
                      grid=(6,7))
    pyplot.savefig("interact_off_quercetin_above_sictrl_ubound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment': 'Quercetin'},
                      ref_val=interact_off_quer_lbound,
                      grid=(6,7))
    pyplot.savefig("interact_off_quercetin_below_sictrl_lbound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_effect'],
                      coords={'treatment': ['DMSO', 'Fisetin', 'Quercetin'],
                              'gene': 'siCtrl'})
    pyplot.savefig("interact_effect_no_silence_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    interact_effect_tab = az.summary(itrace, ['interact_effect'])
    where_queronly = interact_effect_tab.index == 'interact_effect[Quercetin, siCtrl]'
    interact_effect_quer_only = interact_effect_tab.loc[where_queronly]
    interact_effect_quer_lbound = interact_effect_quer_only.iloc[0]['hdi_3%']
    interact_effect_quer_hbound = interact_effect_quer_only.iloc[0]['hdi_97%']
    
    az.plot_posterior(itrace,
                      ['interact_effect'],
                      coords={'treatment': 'Quercetin'},
                      ref_val=interact_effect_quer_lbound,
                      grid=(6,7))
    pyplot.savefig("interact_effect_quercetin_below_sictrl_lbound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_effect'],
                      coords={'treatment': 'Quercetin'},
                      ref_val=interact_effect_quer_hbound,
                      grid=(6,7))
    pyplot.savefig("interact_effect_quercetin_above_sictrl_ubound.svg", bbox_inches="tight")
    pyplot.show()
    
    axes_ = az.plot_forest(itrace, var_names=['interact_effect'], coords={'treatment': 'DMSO'})
    axes_[0].set_title("94.0% HDI: DMSO")
    pyplot.savefig("interact_effect_dmso_posterior_forest.svg", bbox_inches="tight")
    pyplot.show()
    
    axes_ = az.plot_forest(itrace, var_names=['interact_effect'], coords={'treatment': 'Fisetin'})
    axes_[0].set_title("94.0% HDI: Fisetin")
    pyplot.savefig("interact_effect_fisetin_posterior_forest.svg", bbox_inches="tight")
    pyplot.show()
    
    axes_ = az.plot_forest(itrace, var_names=['interact_effect'], coords={'treatment': 'Quercetin'})
    axes_[0].set_title("94% HDI: Quercetin")
    pyplot.savefig("interact_effect_quercetin_posterior_forest.svg", bbox_inches="tight")
    pyplot.show()
    
    
    where_dmsoonly = interact_off_tab.index == 'interact_off[DMSO, siCtrl]'
    interact_dmso_only = interact_off_tab.loc[where_dmsoonly]
    interact_off_dmso_lbound = interact_dmso_only.iloc[0]['hdi_3%']
    interact_off_dmso_hbound = interact_dmso_only.iloc[0]['hdi_97%']
    az.plot_posterior(itrace, 
                      ['interact_off'],
                      coords={'treatment': 'DMSO'}, 
                      grid=(6,7),
                      ref_val=interact_off_dmso_hbound)
    pyplot.savefig("interact_off_dmso_posterior_above_sictrl_ubound.svg", bbox_inches="tight")
    pyplot.show()
    
    az.plot_posterior(itrace,
                      ['interact_off'],
                      coords={'treatment':'DMSO'},
                      grid=(6,7),
                      ref_val=interact_off_dmso_lbound)
    pyplot.savefig("interact_off_dmso_posterior_below_lbound.svg", bbox_inches='tight')
    pyplot.show()
    
    
    gene_const_sum = az.summary(itrace, ['gene_const'])
    gene_const_control_lowbound = gene_const_sum.loc['gene_const[siCtrl]', 'hdi_3%']
    gene_const_prob_below = (itrace.posterior['gene_const'] < 
                             gene_const_control_lowbound).mean(('chain','draw'))
    gene_const_table = gene_const_prob_below.to_dataframe()
    gene_const_table = gene_const_table.rename(columns={'gene_const': 'prob_below_ctrl_hdi'})
    
    fisetin_interact_off_sum = az.summary(itrace, 
                                          ['interact_off'], 
                                          coords={'treatment': 'Fisetin'})
    fis_interact_off_highbound = fisetin_interact_off_sum.loc['interact_off[siCtrl]', 'hdi_97%']
    fis_interact_off_above = (itrace.posterior['interact_off'] > 
                              fis_interact_off_highbound).mean(('chain', 'draw')).sel({'treatment': 'Fisetin'})
    fis_interact_off_table = fis_interact_off_above.to_dataframe()
    del fis_interact_off_table['treatment']
    fis_interact_off_table = fis_interact_off_table.rename(columns={'interact_off': 'fis_prob_dstep_mag_smaller_hdi'})
    
    quer_interact_off_sum = az.summary(itrace,
                                       ['interact_off'],
                                       coords={'treatment': 'Quercetin'})
    quer_interact_off_highbound = quer_interact_off_sum.loc['interact_off[siCtrl]', 'hdi_97%']
    quer_interact_off_above = (itrace.posterior['interact_off'] >
                               quer_interact_off_highbound).mean(('chain','draw')).sel({'treatment': 'Quercetin'})
    quer_interact_off_table = quer_interact_off_above.to_dataframe()
    del quer_interact_off_table['treatment']
    quer_interact_off_table = quer_interact_off_table.rename(columns={'interact_off': 'quer_prob_dstep_mag_smaller_hdi'})
    
    all_deciders = gene_const_table.join(fis_interact_off_table).join(quer_interact_off_table)
    all_deciders.to_csv("decision_probs.csv")
    
    sb.barplot(all_deciders, x='gene', y='prob_below_ctrl_hdi')
    pyplot.ylabel("probability of gene being pro-survival")
    pyplot.xticks(rotation=90)
    pyplot.title("pro-survival probabilities")
    pyplot.axhline(0.9)
    pyplot.savefig("pro_survival_probs.svg", bbox_inches="tight")
    pyplot.show()
    
    sb.barplot(all_deciders, x='gene', y='fis_prob_dstep_mag_smaller_hdi')
    pyplot.ylabel("probability of gene contributing to fisetin activity")
    pyplot.xticks(rotation=90)
    pyplot.title("fisetin involvement probabilities")
    pyplot.axhline(0.9)
    pyplot.savefig("fisetin_probs.svg", bbox_inches="tight")
    pyplot.show()
    
    sb.barplot(all_deciders, x='gene', y='quer_prob_dstep_mag_smaller_hdi')
    pyplot.ylabel("probability of gene contributing to quercetin activity")
    pyplot.xticks(rotation=90)
    pyplot.title("quercetin involvement probabilities")
    pyplot.axhline(0.9)
    pyplot.savefig("quercetin_probs.svg", bbox_inches="tight")
    pyplot.show()
    
    
    all_deciders['survival'] = all_deciders['prob_below_ctrl_hdi'] > 0.9
    all_deciders['quercetin'] = all_deciders['fis_prob_dstep_mag_smaller_hdi'] > 0.9
    all_deciders['fisetin'] = all_deciders['quer_prob_dstep_mag_smaller_hdi'] > 0.9
    
    surv_genes = set(all_deciders.loc[all_deciders['survival']].index)
    quer_genes = set(all_deciders.loc[all_deciders['quercetin']].index)
    fis_genes = set(all_deciders.loc[all_deciders['fisetin']].index)
    
    v3 = venn.venn3([surv_genes, quer_genes, fis_genes],
                    set_labels=("survival", "quercetin", "fisetin"),
                    layout_algorithm=venn.layout.venn3.DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1))
                    )
    pyplot.savefig("overlap_size.svg", bbox_inches="tight")
    pyplot.show()
    
    
    with pyplot.rc_context({'figure.figsize': [16,16]}):
        v3 = venn.venn3([surv_genes, quer_genes, fis_genes],
                        set_labels=("survival", "quercetin", "fisetin")#,
                        #layout_algorithm=venn.layout.venn3.DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1))
                        )
        v3.get_label_by_id('111').set_text('\n'.join(surv_genes&quer_genes&fis_genes))
        v3.get_label_by_id('100').set_text('\n'.join(surv_genes-(quer_genes|fis_genes)))
        v3.get_label_by_id('011').set_text('\n'.join((quer_genes&fis_genes)-surv_genes))
        v3.get_label_by_id('010').set_text('\n'.join(quer_genes-(surv_genes|fis_genes)))
        surv_quer = (quer_genes&surv_genes)-fis_genes
        sq_pairs = itertools.batched(surv_quer, 2)
        sq = map(lambda p: ", ".join(p), sq_pairs)
        sq_name = "\n".join(sq)
        v3.get_label_by_id('110').set_text(sq_name)
        pyplot.savefig("overlap_members.svg", bbox_inches="tight")
        pyplot.show()
    
    
    