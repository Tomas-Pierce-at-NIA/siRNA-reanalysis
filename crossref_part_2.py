# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:15:19 2025

@author: piercetf
"""

import polars as pl
import seaborn as sb
from matplotlib import pyplot

nparc_cross = "sirna_nparc_crossref.csv"
nctab = pl.read_csv(nparc_cross)
