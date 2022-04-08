rm(list = ls())
library(tidyverse)
library(rlang)


# Load Data ---------------------------------------------------------------
data <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/cnn_cv_auc.csv")

auc_plot <- function(data, auc_col, auc_lab){
  "
  Make a grouped bar plot of the auc_col. The groups are different redundancy levels.
  "
  data %>%
    mutate(label = as_factor(label)) %>%
    ggplot(mapping = aes(x = fct_reorder(peptide, desc(count)), y = !!sym(auc_col), fill = label))+
    geom_bar(stat = "identity", position = "dodge")+
    geom_text(aes(label = round(!!sym(auc_col), 2)), 
              position = position_dodge(width = 0.9),
              size=2, 
              angle=90,
              hjust = 1.5)+
    labs(fill = "Redundancy", y = auc_lab, x = "")+
    guides(fill = guide_legend(reverse = TRUE))+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 90))+
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10))
  
}

auc_plot(data, auc_col = "auc", auc_lab = "AUC")
auc_plot(data, auc_col = "auc_0.1", auc_lab = "AUC 0.1")
auc_plot(data, auc_col = "auc_swapped", auc_lab = "AUC Swapped")
auc_plot(data, auc_col = "auc_swapped_0.1", auc_lab = "AUC Swapped 0.1")
auc_plot(data, auc_col = "auc_10x", auc_lab = "AUC 10x")
auc_plot(data, auc_col = "auc_10x_0.1", auc_lab = "AUC 10x 0.1")

