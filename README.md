# Prediction TCR:pMHC interactions using Deep Learning

The interaction between T cell receptors (TCRs) and peptide-major histocompatibility complex (pMHC) is vital for a functioning immune system, and understanding this interaction has potential applications within vaccine development and cancer treatments.

## Data
The data consists of the sequence of peptide-MHC complexes restricted to the HLA-A*02:01 and the T cell receptor sequence. Multiple common peptide epitopes. Alpha and beta chains of the variable domain of the TCR.

## Project goal
The overall goal of this project is to develop deep learning methods to predict interactions between TCRs and pMHC.
However in the process multiple sub-goals have been established.

* Understand which parts of the dataset has a significant meaning for predicting the interaction
  * Here different experiments has been done to investigate effects of different CDR regions and alpha vs beta chains.
  * Use simple and learned attention mechanisms to locate specific subregions within the sequence with higher importance than others.
* Determine what network architectures works best for modelling of TCR pMHC interactions
* See if pan-specific models can outperform peptide-specific models.
