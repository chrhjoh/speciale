rm(list = ls())
library(tidyverse)
library(rlang)
library(ggridges)


# Load Data ---------------------------------------------------------------
cnn_pep_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/cnn_cv_auc.csv")
lstm_tcr_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_positive_allpep_auc.csv")
scores_data <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_attention_scores.csv",
                        col_names = c("ID", "peptide", "origin", "score", "label"))


# Function for plotting
auc_plot <- function(data, auc_col, auc_lab){
  "
  Make a grouped bar plot of the auc_col. The groups are different redundancy levels.
  "
  data %>%
    mutate(redundancy = as_factor(redundancy)) %>%
    ggplot(mapping = aes(x = fct_reorder(peptide, desc(count)), y = !!sym(auc_col), fill = redundancy))+
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


# AUC bar plots -----------------------------------------------------------
auc_plot(cnn_pep_auc, auc_col = "auc", auc_lab = "AUC")
auc_plot(cnn_pep_auc, auc_col = "auc_0.1", auc_lab = "AUC 0.1")
auc_plot(cnn_pep_auc, auc_col = "auc_swapped", auc_lab = "AUC Swapped")
auc_plot(cnn_pep_auc, auc_col = "auc_swapped_0.1", auc_lab = "AUC Swapped 0.1")
auc_plot(cnn_pep_auc, auc_col = "auc_10x", auc_lab = "AUC 10x")
auc_plot(cnn_pep_auc, auc_col = "auc_10x_0.1", auc_lab = "AUC 10x 0.1")

# AUC of positive tcrs against swapped negatives --------------------------

lstm_tcr_auc <- lstm_tcr_auc %>%
  group_by(peptide) %>%
  add_count(peptide, name = "counts") %>%
  filter(counts > 2) %>%
  mutate(peptide = as_factor(peptide))

# generate strings for labels
label_strings <- lstm_tcr_auc %>%
  summarise(counts = n()) %>%
  mutate(label = str_c(peptide, " (", counts, ")")) %>%
  arrange(desc(counts))

lstm_tcr_auc %>%
  ggplot(mapping = aes(y = fct_reorder(peptide, desc(counts)), x = auc, fill = peptide))+
  geom_density_ridges(alpha=0.4)+
  scale_x_continuous(limits = c(0, 1))+
  scale_y_discrete(labels = label_strings$label)+
  labs(y="", x="AUC")+
  theme_bw()+
  theme(legend.position = "none")
# Note that NLV would often predict SL peptide higher instead. LL is probably due to very low counts


# plots of score distributions per peptide and origin ---------------------

scores_data <- scores_data %>%
  group_by(peptide) %>%
  mutate(origin = str_replace(origin, "_[:digit:]+", ""),
         peptide = as_factor(peptide),
         score = as.numeric(score)) %>%
  add_count(peptide) %>%
  filter(n > 40,
         score < 1)

scores_data %>%
  ggplot(mapping = aes(x = fct_reorder(peptide, desc(n)), y = score, fill = origin))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle = 90))+
  labs(x = "Peptide")

