# ================================================
# 1. Load necessary packages
# ================================================
install.packages(c("readxl", "dplyr", "ggplot2", "tidyr", "emmeans"))
library(readxl)
library(dplyr)
library(ggplot2)
library(tidyr)
library(emmeans)

# ================================================
# 2. Read the Excel file
# ================================================
# Replace this with the actual filename
df <- read_excel("db2.xlsx")

# ================================================
# 3. Preprocess and clean the data
# ================================================
df <- df %>%
  mutate(
    luminiscence = as.numeric(luminiscence),  # convert to numeric if needed
    siRNA = factor(siRNA),
    treatment = factor(treatment, levels = c("DMSO", "Fisetin", "Quercetin")),
    time = factor(time, levels = c("T0", "T1", "T2", "T3")),
    rep = as.factor(rep),
    logLum = log2(luminiscence + 1)           # log transform to reduce skew
  )

# Check number of missing values
sum(is.na(df$luminiscence))

# ================================================
# 4. OBJECTIVE 1 – siRNA effect without drug
# ================================================
df_dms <- df %>% filter(treatment == "DMSO")

# Plot: Boxplots of siRNA viability per timepoint (stacked vertically)
ggplot(df_dms, aes(x = siRNA, y = logLum)) +
  geom_boxplot() +
  facet_wrap(~time, ncol = 1) +
  labs(
    title = "Viability under DMSO only (no drug)",
    y = "log2(Luminescence + 1)",
    x = "siRNA"
  ) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# ANOVA to test siRNA effect on viability (under DMSO)
model_dms <- aov(logLum ~ siRNA * time, data = df_dms)
summary(model_dms)

# Post-hoc comparisons per time point
emmeans(model_dms, pairwise ~ siRNA | time)

# ================================================
# 5. OBJECTIVE 2 – Does siRNA mimic or block drug action?
# ================================================

# Filter data for relevant comparisons:
# - siCtrl + DMSO
# - siCtrl + Fisetin or Quercetin
# - siRNA + DMSO
# - siRNA + Fisetin or Quercetin
df_comp <- df %>%
  filter(
    (siRNA == "siCtrl" & treatment %in% c("DMSO", "Fisetin", "Quercetin")) |
      (siRNA != "siCtrl" & treatment %in% c("DMSO", "Fisetin", "Quercetin"))
  )

# Plot: Viability over time by treatment and siRNA
ggplot(df_comp, aes(x = time, y = logLum, color = treatment, group = treatment)) +
  stat_summary(fun = mean, geom = "line", linewidth = 0.5) +
  stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2) +
  facet_wrap(~siRNA) +
  labs(
    title = "Viability by treatment and siRNA",
    y = "log2(Luminescence + 1)",
    x = "Time"
  ) +
  theme_minimal()

# ANOVA: interaction model to detect if siRNA blocks drug effect
model_all <- aov(logLum ~ siRNA * treatment * time, data = df_comp)
summary(model_all)

# Post-hoc test: compare treatment effects within each siRNA and time
emmeans(model_all, pairwise ~ treatment | siRNA * time)

# ================================================
# 6. EXTRA: Visual comparison of siRNA+DMSO vs siCtrl+Drug
# ================================================
# Focus on timepoints where drug effect is expected (T2, T3)
df_focus <- df %>%
  filter(time %in% c("T2", "T3")) %>%
  filter(
    (siRNA == "siCtrl" & treatment %in% c("DMSO", "Fisetin", "Quercetin")) |
      (treatment == "DMSO")
  )

# Plot: stacked comparison of siRNA + DMSO vs siCtrl + Drug
ggplot(df_focus, aes(x = interaction(siRNA, treatment), y = logLum)) +
  geom_boxplot() +
  facet_wrap(~time, ncol = 1) +
  labs(
    title = "Does siRNA mimic the drug effect?",
    x = "Condition (siRNA + Treatment)",
    y = "log2(Luminescence + 1)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

