rm(list= ls())
library(tidyverse)
library(RColorBrewer)
library(rlang)
library(docstring)
library(patchwork)


# Functions for wrangling and plotting ------------------------------------

wrangle_feature <- function(data, origins, feature, direction){
  #' Takes a tibble and wrangles it.
  #' Subset columns to only contain meta data and desired features
  #' Either CDR3a or CDR3b for a given direction.
  #' Then converts to a long format that can used to plot

  # Extract meta data columns
  metacols <- c("ID", "partition", "pep", "cdr3a", "cdr3b", "origin", "label")
  meta_data <- data %>%
    select(metacols)
  
  # Subset data to only contain one type of attention
  data %>%
    select(str_c(feature, "_", direction) %>% starts_with(), "ID") %>%
    inner_join(meta_data, by = "ID") %>%
    
    add_count(pep, name = "pep_count") %>%
    filter(str_starts(origin, origins)) %>%
    filter(pep_count > 30) %>%
    
    mutate(sort_len = !!as.name(feature) %>% str_length()) %>%
    arrange(pep_count, label, sort_len) %>%
    rowid_to_column("plot_id") %>%
      
    pivot_longer(cols = str_c(feature, "_") %>% starts_with(),
                 names_to = "position",
                 values_to = "attention") %>%
                   
    # Grab the position from the name and turn into numeric
    mutate(position = str_replace(position, str_c(feature,"_",direction,"_([:digit:]+)"), "\\1") %>% 
                      as.numeric())
}

attention_plot_cdrs <- function(df, len){
  #' Plots a tibble wrangled by wrangled data to give an attention plot

  df %>%
  ggplot(mapping = aes(x = position, y = plot_id, fill = attention)) +
  geom_rug(aes(color = pep), sides = "l")+
  geom_tile(show.legend = FALSE)+
  scale_colour_gradient(
    low = "white",
    high = "black",
    guide = "colourbar",
    aesthetics = "fill"
  )+
  scale_x_continuous(breaks = seq(1, len, by = 1))+
  theme_minimal()+
  labs(color="")+
  theme(axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.text.y = element_blank(),
        panel.background = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        legend.position = "bottom")
}

attention_plot_tcrs <- function(df, anno){
  #' Plots a tibble wrangled by wrangled data to give an attention plot
  anno["obs"] = max(df$plot_id)
  df %>%
    ggplot(mapping = aes(x = position,
                         y = plot_id,
                         fill = attention)) +
    geom_rug(aes(color = pep), sides = "l")+
    geom_tile(show.legend = FALSE)+
    annotate("rect",
             ymin = c(0,0,0),
             ymax = c(anno$obs,anno$obs, anno$obs),
             xmin = c(anno$cdr1_start, anno$cdr2_start, anno$cdr3_start),
             xmax = c(anno$cdr1_end, anno$cdr2_end, anno$cdr3_end),
             alpha = 0.3,
             fill = "darkturquoise")+
    scale_colour_gradient(
      low = "white",
      high = "black",
      guide = "colourbar",
      aesthetics = "fill"
    )+
    theme_minimal()+
    labs(color="", x="")+
    theme(axis.line.y = element_blank(),
          axis.ticks.y = element_blank(),
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.text.x = element_blank(),
          axis.title.x = element_blank(),
          panel.background = element_blank(),
          panel.grid.minor = element_blank(),
          panel.grid.major = element_blank(),
          legend.position = "bottom")
}

# Load Data ---------------------------------------------------------------
data_tcrs <- read_csv('/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_attention_partition5_tcrs_saved.csv')
data_cdrs <- read_csv('/Users/christianjohansen/Desktop/speciale/modeling/results/lstm_attention_partition5_98_saved.csv')
all_data <- read_csv('/Users/christianjohansen/Desktop/speciale/modeling/data/datasets/train_data_all.csv')
# Wrangle Data -----------------------------------------

cdr3b_data_f <- wrangle_feature(data_cdrs, "positive", "cdr3b", "forward")
cdr3b_data_f_neg <- wrangle_feature(data_cdrs, "10x", "cdr3b", "forward")
cdr3a_data_f <- wrangle_feature(data_cdrs, "positive", "cdr3a", "forward")
cdr3a_data_f_neg <- wrangle_feature(data_cdrs, "10x", "cdr3a", "forward")
cdr3a_data_r <- wrangle_feature(data_cdrs, "positive", "cdr3a", "reverse")

tcrb_data_f <- wrangle_feature(data_tcrs, "positive", "cdr3b", "forward")
tcra_data_f <- wrangle_feature(data_tcrs, "positive", "cdr3a", "forward")


# Calculate fraction of attention in CDRs ---------------------------------
anno_alpha = list("cdr1_start" = 20, "cdr1_end" = 27, 
                  "cdr2_start" = 45, "cdr2_end" = 54,
                  "cdr3_start" = 84, "cdr3_end" = 101)
anno_beta = list("cdr1_start" = 22, "cdr1_end" = 28, 
                 "cdr2_start" = 47, "cdr2_end" = 55,
                 "cdr3_start" = 89, "cdr3_end" = 107)

estimate_used_attention <- function(low, high, att){
  att %>%
    group_by(position) %>%
    summarise(attention = mean(attention)) %>%
    filter(low < position & position < high) %>%
    summarise(predicted = sum(attention),
              expected = (high - low) / max(att$position) )
}


estimate_used_attention(anno_alpha$cdr3_start, anno_alpha$cdr3_end, tcra_data_f)
estimate_used_attention(anno_beta$cdr3_start, anno_beta$cdr3_end, tcrb_data_f)

# Plot Data ---------------------------------------------------------------
plot1 <- attention_plot_tcrs(tcra_data_f, anno_alpha)
plot2 <- attention_plot_tcrs(tcrb_data_f, anno_beta)
plot1 + plot2 + plot_layout(guides = "collect") + plot_annotation(tag_levels = 'A')& theme(legend.position = "bottom",
                                                                                           plot.tag = element_text(face = "bold"))
plot1 <- attention_plot_cdrs(cdr3a_data_f, 15)
plot2 <- attention_plot_cdrs(cdr3a_data_f_neg, 15)
(plot1 + plot2) + plot_layout(guides = "collect") + plot_annotation(tag_levels = 'A')& theme(legend.position = "bottom",
                                                                                           plot.tag = element_text(face = "bold"))
attention_plot_cdrs(cdr3a_data_r, 15)


plot3 <- attention_plot_cdrs(cdr3b_data_f, 17)
plot4 <-attention_plot_cdrs(cdr3b_data_f_neg, 17)
(plot1 + plot2) / (plot3 + plot4) + plot_layout(guides = "collect") + plot_annotation(tag_levels = 'A')& theme(legend.position = "bottom",
                                                                                           plot.tag = element_text(face = "bold"))

# Fraction of attention put into the annotated areas on the TCRs

# Wrangle data for CDR3 length plot
all_cdrs <- all_data %>%
  select(cdr3a, cdr3b, label) %>%
  pivot_longer(cols = c(cdr3a, cdr3b),
               names_to = "type",
               values_to = "sequence") %>%
  mutate(cdr_len = str_length(sequence) %>% as_factor(),
         label = if_else(label==1, true = "Positive", false = "Negative") %>% as_factor())


# Plots to show why the attention is so alike for observations on the cdr3b
# Just a stratified count of the cdr lengths
cdr_plot_data %>%
  filter(type == "cdr3b") %>%
  ggplot(mapping = aes(x = cdr_len, fill = label))+
  geom_bar()+
  facet_wrap(vars(label))+
  labs(x="CDR3b length")+
  theme_bw()+
  theme(legend.position="none")

cdr_plot_data %>%
  filter(type == "cdr3a") %>%
  ggplot(mapping = aes(x=cdr_len, fill=label))+
  geom_bar()+
  facet_wrap(vars(label))+
  labs(x="CDR3a length")+
  theme_bw()+
  theme(legend.position="none")

  

