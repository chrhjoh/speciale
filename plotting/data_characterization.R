rm(list = ls())
library(tidyverse)
library(patchwork)
data <- read_csv("/Users/christianjohansen/Desktop/speciale/modeling/data/datasets/train_data_all.csv")



# CDR 3 length plot -------------------------------------------------------
cdr3b_lengths <- data %>%
  mutate(cdr3_len = str_length(cdr3b),
         label = case_when(origin == "positive" ~ "Positive", 
                           origin == "10x" ~ "Negative"))%>%
  filter(cdr3_len >= 8) %>%
  drop_na() %>% # drop swapped
  ggplot(mapping = aes(x = cdr3_len))+
  geom_histogram()+
  facet_wrap(~label)+
  theme_bw()+
  labs(x = "CDR3b length", y="Number of observations")+
  scale_x_continuous(breaks = seq(8,19,2))


cdr3a_lengths <- data %>%
  mutate(cdr3_len = str_length(cdr3a),
         label = case_when(origin == "positive" ~ "Positive", 
                           origin == "10x" ~ "Negative"))%>%
  filter(cdr3_len >= 8) %>%
  drop_na() %>% # drop swapped
  ggplot(mapping = aes(x = cdr3_len))+
  geom_histogram()+
  facet_wrap(~label)+
  theme_bw()+
  labs(x = "CDR3a length", y="Number of observations")


cdr3a_lengths + cdr3b_lengths


# Number of observations per peptide and origin ---------------------------
selected_peptides <- data %>%
  group_by(pep) %>%
  add_count(pep, name = "n") %>%
  slice_head(n=1) %>%
  ungroup() %>%
  arrange(desc(n)) %>%
  slice_head(n=10) %>%
  select(pep)


data %>%
  mutate(origin = if_else(startsWith(origin, "swapped"), "swapped", origin) %>% factor(levels = c("10x", "swapped", "positive"))) %>%
  add_count(pep, name="n") %>%
  filter(pep %in% selected_peptides$pep) %>%
  ggplot(mapping = aes(x = fct_reorder(pep, desc(n)), fill = origin ))+
  geom_bar(position = "dodge")+
  labs(x = "Peptide", y = "Number of Observations", fill = "Origin")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


  





















