rm(list =ls())
library(tidyverse)
library(rlang)
library(ggridges)


# Load Data ---------------------------------------------------------------
auc_per_peptide_cnn <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/cnn_cv_auc.csv")
auc_per_peptide_lstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_cv_auc.csv")
auc_per_peptide_attlstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/attlstm_cv_auc.csv")
auc_per_peptide_embattlstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/embattlstm_cv_auc.csv")


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

auc_line_plot <- function(data, auc_col, auc_lab){
  "
  Make a grouped bar plot of the auc_col. The groups are different redundancy levels.
  "
  data %>%
    ggplot(mapping = aes(x = redundancy, y = !!sym(auc_col), color = peptide))+
    geom_line()+
    labs(fill = "Redundancy", y = auc_lab, x = "")+
    guides(fill = guide_legend(reverse = TRUE))+
    theme_bw()+
    theme(axis.text.x = element_text(angle = 90))+
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10))
}


# AUC bar plots -----------------------------------------------------------
auc_plot(auc_per_peptide_attlstm, auc_col = "auc", auc_lab = "AUC")
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_0.1", auc_lab = "AUC 0.1")
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_swapped", auc_lab = "AUC Swapped")
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_swapped_0.1", auc_lab = "AUC Swapped 0.1")
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_10x", auc_lab = "AUC 10x")
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_10x_0.1", auc_lab = "AUC 10x 0.1")


# AUC plot of total performance for all partitions ------------------------
auc_per_peptide_cnn <- auc_per_peptide_cnn %>% mutate(model = "CNN")
auc_per_peptide_lstm <- auc_per_peptide_lstm %>% mutate(model = "LSTM")
auc_per_peptide_attlstm <- auc_per_peptide_attlstm %>% mutate(model = "Attention LSTM")
auc_per_peptide_embattlstm <- auc_per_peptide_embattlstm %>% mutate(model = "Embedded Attention LSTM")

# Have to combine all the total results for this, 
# I have just added the baseline results manually
bind_rows(auc_per_peptide_cnn,
            auc_per_peptide_lstm,
            auc_per_peptide_attlstm,
            auc_per_peptide_embattlstm
            ) %>%
  filter(peptide == "total") %>%
  select(auc, redundancy, model) %>%
  bind_rows(tibble(auc = c(0.85, 0.84, 0.78, 0.66), 
                   redundancy = c(1, 0.98, 0.95, 0.90), 
                   model=rep("baseline", 4))) %>%
  ggplot(mapping = aes(x = redundancy, y = auc, color = model))+
  geom_line()



# subsample AUCs with pretraining ----------------------------------------------------------

subsample_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_subsample_auc.csv")
subsample_pre100_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL100pretrain_subsample_auc.csv")
subsample_pre50_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL50pretrain_subsample_auc.csv")
subsample_pre20_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL20pretrain_subsample_auc.csv")
subsample_pre10_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL10pretrain_subsample_auc.csv")
subsample_pre5_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL5pretrain_subsample_auc.csv")

meta_data <- read_csv("/Users/christianjohansen/Desktop/speciale/processing/data/datasets/metadata_subsamples.csv") %>%
              mutate(sampling = sampling / 100) # The AUCs has fractions instead of percentage

# For now just quickly display subsample curve for the total performance

subsample_auc <- subsample_auc %>% mutate(pretrain = 0)
subsample_pre100_auc <- subsample_pre100_auc %>% mutate(pretrain = 1)
subsample_pre50_auc <- subsample_pre50_auc %>% mutate(pretrain = 0.5)
subsample_pre20_auc <- subsample_pre20_auc %>% mutate(pretrain = 0.2)
subsample_pre10_auc <- subsample_pre10_auc %>% mutate(pretrain = 0.1)
subsample_pre5_auc <- subsample_pre5_auc %>% mutate(pretrain = 0.05)



bind_rows(subsample_auc, subsample_pre100_auc, subsample_pre50_auc,
          subsample_pre20_auc, subsample_pre10_auc, subsample_pre5_auc) %>%
  mutate(pretrain = as_factor(pretrain),
         peptide = factor(peptide, levels = c("total", "GLCTLVAML", "NLVPMVATV", "FLYALALLL"))) %>%
  filter(peptide != "LLFGYPVYV" & peptide != "RTLNAWVKV"  & redundancy != 0.05 & redundancy != 0.1) %>%
  ggplot(mapping = aes(x = redundancy, y = auc, color = pretrain))+
  geom_line()+
  facet_wrap(~peptide)+
  labs(x = "Sample Fraction", color="GIL pretraining fraction")+
  guides(color = guide_legend(reverse = TRUE))



# Pretraining on number of positives --------------------------------------

bind_rows(subsample_auc, subsample_pre100_auc, subsample_pre50_auc,
          subsample_pre20_auc, subsample_pre10_auc, subsample_pre5_auc) %>%
  inner_join(meta_data, by = c("redundancy" = "sampling", "peptide" = "pep")) %>%
  filter(origin == "positive" & redundancy != 0.05 & redundancy != 0.1 & (pretrain == 1 | pretrain == 0)) %>%
  mutate(pretrain = as_factor(pretrain),
         peptide = factor(peptide, levels = c("total", "GLCTLVAML", "NLVPMVATV", "FLYALALLL"))) %>%
  drop_na() %>%
  ggplot(mapping = aes(x = counts, y = auc_10x, color = peptide))+
  geom_line()+
  facet_wrap(~pretrain)+
  labs(x = "Number of positives", color="GIL pretraining fraction")+
  guides(color = guide_legend(reverse = TRUE))


# Subsample peptide specific vs pan ---------------------------------------
pan_gil_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmpan_GIL_auc.csv")
specific_gil_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmsingle_GIL_auc.csv")

pan_gil_auc <- pan_gil_auc %>% mutate(model = "pan")
specific_gil_auc <- specific_gil_auc %>% mutate(model = "specific")

bind_rows(pan_gil_auc, specific_gil_auc) %>%
  filter(peptide == "GILGFVFTL") %>%
  ggplot(mapping = aes(x = n_positives, y = auc, color = model))+
  geom_line()+
  labs(x = "Number of positives", color="GILGFVFTL model")+
  guides(color = guide_legend(reverse = TRUE))




# AUC of positive tcrs against swapped negatives --------------------------
auc_per_tcr <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_positive_allpep_auc.csv")

auc_per_tcr <- auc_per_tcr %>%
  group_by(peptide) %>%
  add_count(peptide, name = "counts") %>%
  filter(counts > 2) %>%
  mutate(peptide = as_factor(peptide))

# generate strings for labels
label_strings <- auc_per_tcr %>%
  summarise(counts = n()) %>%
  mutate(label = str_c(peptide, " (", counts, ")")) %>%
  arrange(desc(counts))

auc_per_tcr %>%
  ggplot(mapping = aes(y = fct_reorder(peptide, desc(counts)), x = auc, fill = peptide))+
  geom_boxplot(alpha=0.4)+
  scale_x_continuous(limits = c(0, 1))+
  scale_y_discrete(labels = label_strings$label)+
  labs(y="", x="AUC")+
  theme_bw()+
  theme(legend.position = "none")
# Note that NLV would often predict SL peptide higher instead. LL is probably due to very low counts
# Decides that for GLC or GIL whether the model believes this is this peptide or not. For other peptides not so much

# plots of score distributions per peptide and origin ---------------------

scores_data <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_attention_scores.csv",
                        col_names = c("ID", "peptide", "origin", "score", "label"))

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

