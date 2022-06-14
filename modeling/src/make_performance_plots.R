rm(list =ls())
library(tidyverse)
library(rlang)
library(ggridges)
library(patchwork)
library(RColorBrewer)


# Load Data ---------------------------------------------------------------
auc_per_peptide_cnn <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/cnn_cv_auc.csv")
auc_per_peptide_lstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_cv_auc.csv")
auc_per_peptide_attlstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/attlstm_cv_auc.csv")
auc_per_peptide_embattlstm <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/embattlstm_cv_auc.csv")


meta_data <- read_csv("/Users/christianjohansen/Desktop/speciale/processing/data/datasets/metadata_subsamples.csv") %>%
  mutate(sampling = sampling / 100) # The AUCs has fractions instead of percentage
label_strings <- meta_data %>%
  mutate(label = str_c(pep, " (", counts, ")")) %>%
  filter(origin == "positive", sampling == 1) %>%
  arrange(desc(counts)) %>%
  bind_rows(c("pep"="total", "origin"="positive",  "label"="Total"))

custom_labels <- function(breaks){
  labels <- label_strings[match(breaks, label_strings$pep),]
  return(labels$label)
  
} 

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
    theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))+
    scale_y_continuous(breaks = scales::pretty_breaks(n = 10))+
    scale_x_discrete(labels=custom_labels)
  
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
auc_plot(auc_per_peptide_attlstm, auc_col = "auc_10x", auc_lab = "AUC")
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
  geom_line()+
  labs(x = "Redundancy", y = "AUC", color = "Model")+
  theme_bw()



# subsample AUCs with pretraining ----------------------------------------------------------

subsample_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_subsample_auc.csv")
subsample_pre100_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL100pretrain_subsample_auc.csv")
subsample_pre50_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL50pretrain_subsample_auc.csv")
subsample_pre20_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL20pretrain_subsample_auc.csv")
subsample_pre10_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL10pretrain_subsample_auc.csv")
subsample_pre5_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstm_GIL5pretrain_subsample_auc.csv")

# For now just quickly display subsample curve for the total performance

subsample_auc <- subsample_auc %>% mutate(pretrain = 0)
subsample_pre100_auc <- subsample_pre100_auc %>% mutate(pretrain = 1)
subsample_pre50_auc <- subsample_pre50_auc %>% mutate(pretrain = 0.5)
subsample_pre20_auc <- subsample_pre20_auc %>% mutate(pretrain = 0.2)
subsample_pre10_auc <- subsample_pre10_auc %>% mutate(pretrain = 0.1)
subsample_pre5_auc <- subsample_pre5_auc %>% mutate(pretrain = 0.05)

bind_rows(subsample_auc, subsample_pre100_auc, subsample_pre50_auc,) %>%
  mutate(pretrain = as_factor(pretrain),
         peptide = factor(peptide, levels = c("total", "GLCTLVAML", "NLVPMVATV", "FLYALALLL"))) %>%
  inner_join(meta_data, by = c("redundancy" = "sampling", "peptide" = "pep")) %>%
  filter(peptide != "LLFGYPVYV" & peptide != "RTLNAWVKV"  & redundancy != 0.05 & redundancy != 0.1 & origin == "positive") %>%
  ggplot(mapping = aes(x = counts, y = auc, color = pretrain))+
  geom_line()+
  facet_wrap(~peptide)+
  labs(x = "Number of positive TCRs", color="GIL pretraining fraction")+
  guides(color = guide_legend(reverse = TRUE))



# Pretraining on number of positives --------------------------------------

set_labels <- function(var){
  labels <- list("0"="No Pretraining", "1"="Pretrained")
  print(var)
  return(labels[var])
}

bind_rows(subsample_auc, subsample_pre100_auc, subsample_pre50_auc,
          subsample_pre20_auc, subsample_pre10_auc, subsample_pre5_auc) %>%
  inner_join(meta_data, by = c("redundancy" = "sampling", "peptide" = "pep")) %>%
  filter(origin == "positive" & redundancy != 0.05 & redundancy != 0.1 & (pretrain == 1 | pretrain == 0)) %>%
  mutate(pretrain = recode(pretrain, 
                           "0" = "Not Pretrained",
                           "1" = "Pretrained"),
         peptide = factor(peptide, levels = c("total", "GLCTLVAML", "NLVPMVATV", "FLYALALLL"))) %>%
  drop_na() %>%
  ggplot(mapping = aes(x = counts, y = auc_10x, color = peptide))+
  geom_line()+
  facet_wrap(~pretrain)+
  labs(x = "Number of positives", color="GIL pretraining fraction")+
  guides(color = guide_legend(reverse = TRUE))


# Subsample peptide specific vs pan ---------------------------------------
pan_gil_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmpan_GIL_auc.csv") %>%
  filter(peptide == "GILGFVFTL")
specific_gil_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmsingle_GIL_auc.csv") %>%
  filter(peptide == "GILGFVFTL")
pretrain_gil_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/pretrained_attlstmsingle_GIL_auc.csv") %>%
  filter(peptide == "GILGFVFTL")


pan_gil_auc <- pan_gil_auc %>% mutate(model = "pan")
specific_gil_auc <- specific_gil_auc %>% mutate(model = "specific")
pretrain_gil_auc <- pretrain_gil_auc %>% mutate(model = "pretrain")

gil_plot <- bind_rows(pan_gil_auc, specific_gil_auc, pretrain_gil_auc) %>%
  group_by(peptide, n_positives, model) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = model))+
  geom_line()+
  theme_bw()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 5)+
  labs(x = "Number of positives", color="model", y="AUC", title = "Pan or GILGFVFTL specific models")+
  theme(legend.position = "none")

###### GLC
pan_glc_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmpan_GLC_auc.csv") %>%
  filter(peptide == "GLCTLVAML")
specific_glc_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmsingle_GLC_auc.csv") %>%
  filter(peptide == "GLCTLVAML")
pretrain_glc_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/pretrained_attlstmsingle_GLC_auc.csv") %>%
  filter(peptide == "GLCTLVAML")

pan_glc_auc <- pan_glc_auc %>% mutate(model = "pan")
specific_glc_auc <- specific_glc_auc %>% mutate(model = "specific")
pretrain_glc_auc <- pretrain_glc_auc %>% mutate(model = "pretrain")

glc_plot <- bind_rows(pan_glc_auc, specific_glc_auc, pretrain_glc_auc) %>%
  group_by(peptide, n_positives, model) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = model))+
  geom_line()+
  theme_bw()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 1)+
  labs(x = "Number of positives", color=" model", y="AUC", title = "Pan or GLCTLVAML specific models")+
  guides(color = guide_legend(reverse = TRUE))+
  theme(legend.position = "none")
  


#### NLV
pan_nlv_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmpan_NLV_auc.csv") %>%
  filter(peptide == "NLVPMVATV") 
specific_nlv_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/attlstmsingle_NLV_auc.csv") %>%
  filter(peptide == "NLVPMVATV") 
pretrain_nlv_auc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/subsampling/pretrained_attlstmsingle_NLV_auc.csv") %>%
  filter(peptide == "NLVPMVATV") 

pan_nlv_auc <- pan_nlv_auc %>% mutate(model = "Pan")
specific_nlv_auc <- specific_nlv_auc %>% mutate(model = "Specific")
pretrain_nlv_auc <- pretrain_nlv_auc %>% mutate(model = "Pretrain")

nlv_plot <- bind_rows(pan_nlv_auc, specific_nlv_auc, pretrain_nlv_auc) %>%
  group_by(peptide, n_positives, model) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = model))+
  geom_line()+
  theme_bw()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 1)+
  labs(x = "Number of positives", color=" model", y="AUC", title = "Pan or NLVPMVATV specific models")+
  guides(color = guide_legend(reverse = TRUE))

(glc_plot + nlv_plot) / gil_plot + plot_layout(guides = "collect")+ plot_annotation(tag_levels = 'A')& theme(plot.tag = element_text(face = "bold"))

bind_rows(pan_gil_auc, pan_nlv_auc, pan_glc_auc) %>%
  filter(peptide %in% c("NLVPMVATV", "GILGFVFTL", "GLCTLVAML")) %>%
  group_by(peptide, n_positives) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = peptide))+
  geom_line()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 1)+
  theme_bw()+
  labs(x = "Number of positives", color="Peptide", title = "Pan Specific Model", y="AUC")+
  guides(color = guide_legend(reverse = TRUE))

bind_rows(specific_gil_auc, specific_nlv_auc, specific_glc_auc) %>%
  filter(peptide %in% c("NLVPMVATV", "GILGFVFTL", "GLCTLVAML")) %>%
  group_by(peptide, n_positives) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = peptide))+
  geom_line()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 1)+
  theme_bw()+
  labs(x = "Number of positives", color="Peptide", title = "Peptide Specific Models", y="AUC")+
  guides(color = guide_legend(reverse = TRUE))

bind_rows(pretrain_gil_auc, pretrain_nlv_auc, pretrain_glc_auc) %>%
  filter(peptide %in% c("NLVPMVATV", "GILGFVFTL", "GLCTLVAML")) %>%
  group_by(peptide, n_positives) %>%
  summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
  ggplot(mapping = aes(x = n_positives, y = mean_auc, color = peptide))+
  geom_line()+
  geom_errorbar(mapping = aes(ymin = mean_auc - sd_auc, ymax = mean_auc + sd_auc), alpha=0.4, width = 1)+
  theme_bw()+
  labs(x = "Number of positives", color="Peptide", title = "Pretrained Models", y="AUC")+
  guides(color = guide_legend(reverse = TRUE))


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

score_strat <- scores_data %>%
  ggplot(mapping = aes(x = fct_reorder(peptide, desc(n)), y = score, fill = origin))+
  geom_boxplot()+
  theme_bw()+
  labs(fill="Origin", y="Score")+
  theme(axis.text.x = element_text(angle = 45,vjust=1, hjust=1),
        axis.title.x = element_blank(),
        legend.position = "bottom")+
  scale_x_discrete(labels = label_strings$label)




auc_strat <- auc_per_peptide_attlstm %>%
  filter(peptide == "total" & redundancy == 1) %>%
  pivot_longer(cols = starts_with("auc"),
               names_to = "auc_type",
               values_to = "auc") %>%
  filter(auc_type %in% c("auc", "auc_swapped", "auc_10x")) %>%
  mutate(auc_type = recode(auc_type, "auc" = "All", "auc_swapped" = "Postive and Swapped", "auc_10x" = "Positive and 10x")) %>%
  ggplot(mapping=aes(x=peptide,
                     y=auc,
                     fill=auc_type))+
  geom_bar(stat = "identity", position = "dodge",)+
  geom_text(aes(label=round(auc,2)), position=position_dodge(width=0.9), vjust=-0.25)+
  labs(y="AUC", fill="AUC evaluation")+
  theme_bw()+
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = c(0.5,-0.18),
        legend.direction = "vertical")+
  scale_fill_brewer(palette="Paired")
  

auc_strat + score_strat + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(face = "bold"))


# AUC of positive tcrs against swapped negatives --------------------------
tcrauc_pan <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_positive_allpep_auc.csv")
tcrauc_gil <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/GILattlstm_positive_GIL_scores_auc.csv") %>%
  mutate(peptide = "GILGFVFTL",
         model = "Specific Pretrained")
tcrauc_glc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/GLCattlstm_positive_GLC_scores_auc.csv")%>%
  mutate(peptide = "GLCTLVAML",
         model = "Specific Pretrained")
tcrauc_nlv <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/NLVattlstm_positive_NLV_scores_auc.csv") %>%
  mutate(peptide = "NLVPMVATV",
         model = "Specific Pretrained")

tcrauc_pan <- tcrauc_pan %>%
  add_count(peptide, name = "counts") %>%
  rename(Drop = ID, ID = ...1) %>%
  filter(counts > 20) %>%
  mutate(model = "pan")

spec_auc <- bind_rows(tcrauc_pan, tcrauc_gil, tcrauc_glc, tcrauc_nlv) %>%
  mutate(peptide = as_factor(peptide)) %>%
  ggplot(mapping = aes(y = peptide, x = auc, fill = model))+
  geom_boxplot(alpha=0.4)+
  scale_x_continuous(limits = c(0, 1))+
  scale_y_discrete(labels = label_strings$label)+
  scale_fill_viridis_d()+
  labs(y="", x="AUC", fill="Model")+
  theme_bw()

# Note that NLV would often predict SL peptide higher instead. LL is probably due to very low counts
# Decides that for GLC or GIL whether the model believes this is this peptide or not. For other peptides not so much

scores_pan <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_positive_allpep_scores.csv",
                       col_names = c("ID", "peptide", "origin","partition", "score", "label"))
scores_gil <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/GILattlstm_positive_GIL_scores.csv",
                       col_names = c("ID", "peptide", "origin","partition", "score", "label"))
scores_glc <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/GLCattlstm_positive_GLC_scores.csv",
                       col_names = c("ID", "peptide", "origin", "partition","score", "label"))
scores_nlv <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/results/NLVattlstm_positive_NLV_scores.csv",
                       col_names = c("ID", "peptide", "origin", "partition","score", "label"))

specific_scores <- bind_rows(scores_gil, scores_glc, scores_nlv) %>%
  mutate(model = "Specific Pretrained")

positives_spec <- specific_scores %>%
  filter(label == 1) %>%
  select(ID, peptide)

specific_scores <- specific_scores %>%
  inner_join(positives_spec, by= "ID", suffix = c(".swapped", ".tcr_spec"))

positives_pan <- scores_pan %>%
  filter(label == 1 & peptide %in% c("NLVPMVATV", "GILGFVFTL", "GLCTLVAML")) %>%
  select(ID, peptide)

scores_pan <- scores_pan %>%
  mutate(model = "Pan") %>%
  inner_join(positives_pan, by= "ID", suffix = c(".swapped", ".tcr_spec"))

# OBS here the peptide refers to the specificity of the TCR 
# and not what it was paired with to generate the score (for swapped)
spec_scores <- bind_rows(specific_scores, scores_pan) %>%
  ggplot(mapping = aes(x = peptide.tcr_spec, y = score, fill = origin))+
  geom_boxplot(alpha=0.4)+
  facet_wrap(~model)+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 45, hjust=1, vjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_text(vjust=-2))+
  labs(y="Score",
       fill="Origin")+
  scale_x_discrete(labels = label_strings$label)

spec_auc / spec_scores  + plot_annotation(tag_levels = "A") & theme(plot.tag = element_text(face="bold"))

