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
    
    ax.set_title("siRNA Experiment Analysis Results Summary", 
                 fontweight='bold',
                 fontsize=20)
    pyplot.savefig("frequentist_analysis_summary.svg", bbox_inches="tight")
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
    
    quer_diffs = step_itrace.sel(treatment='Quercetin').posterior - step_itrace.sel(treatment='DMSO').posterior
    step_itrace.add_groups({'quer_diffs': quer_diffs})
    day1diffsum = az.summary(step_itrace,
                             ['interact_step1'],
                             group='quer_diffs',
                             coords={'gene': 'siCtrl'})
    diffbounds = day1diffsum.loc['interact_step1', ['hdi_3%', 'hdi_97%']]
    az.plot_posterior(step_itrace,
                      ['interact_step1'],
                      group='quer_diffs',
                      grid=(6,7),
                      ref_val=0.0,
                      rope=diffbounds)
    pyplot.show()
    
    day2diffsum = az.summary(step_itrace,
                             ['interact_step2'],
                             group='fis_diffs',
                             coords={'gene': 'siCtrl'})
    diffbounds = day2diffsum.loc['interact_step2', ['hdi_3%', 'hdi_97%']]
    az.plot_posterior(step_itrace,
                      ['interact_step2'],
                      group='fis_diffs',
                      grid=(6,7),
                      ref_val=0.0,
                      rope=diffbounds)
    pyplot.savefig("day2_fisetin_diff_step_posterior.svg", bbox_inches="tight")
    pyplot.show()
    
    