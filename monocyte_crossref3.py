# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:14:38 2025

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
    
    
    with pm.Model(coords=coords) as model:
        fold_luminescence = pm.Data('fold_luminescence', table['fold_lum'])
        time = pm.Data('time', table['time'])
        gene_idx = pm.Data('gene_idx', table['gene_idnum'])
        treat_idx = pm.Data('treat_idx', table['treat_idnum'])
        
        base_const = pm.Normal('base_const',
                               mu=1,
                               sigma=1)
        gene_const = pm.TruncatedNormal('gene_const',
                                        mu=base_const,
                                        sigma=1,
                                        lower=0,
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
        prior = pm.sample_prior_predictive()
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
    
        

