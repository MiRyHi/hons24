# TCRB analysis clean up from MIXCR output 

#####
# Packages
#####
install.packages()
library(tidyverse)

######
# Import data & clean up
#######
# Setwd to working directory with mixcr output files
setwd()

# Import all files that end in "miXCR clones with PID read cutoff 1
names_exp <- dir()[str_detect(dir(), " miXCR clones summary with PID read cutoff 1.tsv")] %>%
  str_remove(" miXCR clones summary with PID read cutoff 1.tsv")

raw_exp <- dir()[str_detect(dir(), " miXCR clones summary with PID read cutoff 1.tsv")] %>%
  map(read.delim, stringsAsFactors = F)

names(raw_exp) <- names_exp

# Transforming dataframe with the whole TCRB repertoire
# also deleting columns that we don't need, transforming V, D and J gene columns,
exp <- bind_rows(raw_exp, .id = "exp") %>%
  dplyr::select(-ends_with("R1"), 
                -ends_with("R2"), 
                -ends_with("R4"), 
                -ends_with("FR3"),
                -ends_with("ments"),
                -minQualCDR3,
                -clonalSequenceQuality,
                -allCHitsWithScore,
                -clonalSequence, 
                -Well,
                -cloneId,
                -Subject.id) %>%
  mutate(CDR3_length = str_length(aaSeqCDR3)) %>%
  separate(allVHitsWithScore, c("v_gene", NA), sep = "[*]") %>%
  separate(allJHitsWithScore, c("j_gene", NA), sep = "[*]") %>%
  separate(allDHitsWithScore, c("d_gene", NA), sep = "[*]") %>%
  dplyr::select(-starts_with("remove"),
                -starts_with("potential"))


# Excluding out of frame reads, stop codons etc.
# exclude clonesCount<2 - can change if needed
reject_vector <- str_detect(exp$aaSeqCDR3, "_") | 
  str_detect(exp$aaSeqCDR3, "\\*") | 
  str_length(exp$aaSeqCDR3) > 20 | 
  str_length(exp$aaSeqCDR3) < 8 |
  exp$cloneCount < 2

# Making a table/list for all the reject TCRs
exp_rejects <- exp %>%
  filter(reject_vector)
## Have a look and check how many get removed with the filtering

# Filtering out reject TCRs from TCR repertoire
exp_clean <- exp %>%
  filter(!reject_vector)

#######
# Import metadata
#######
# Import metadata vector to get info from each sample
# Write names_exp to a csv and check the 'exp' column matches the 'exp' column 
## in metadata.csv, otherwise the rest of the code will not work.
write.csv(names_exp, "names_exp.csv")

# Import metadata
metadata <- read.csv("metadata.csv", header=TRUE)

# Add metadata vector to TCRB repertoire
exp_final2 <- merge(exp_clean, metadata2, by="exp")

########
# Basic TCR repertoire stats
########
# Functions
entropy <- function(x) {
  H <- vector()
  for (i in 1:length(x)) {
    H[i] <- -x[i]*log(x[i])
  }
  H_norm <- sum(H)/log(length(x))
  H_norm
}

simpsons_index <- function(x) {
  lambda <- vector()
  for (i in 1:length(x)) {
    lambda[i] <- x[i]**2
  }
  lambda <- sum(lambda)
  lambda
}

TCRstats <- exp_final %>%
  group_by(exp, aaSeqCDR3) %>% 
  summarise(PID.count = sum(PID.count), PID.fraction = sum(PID.fraction)) %>% #puts same CDR3s with the different nucleotide sequences together for stats
  group_by(exp) %>%
  summarise(uniqueCDR3 = n(), 
            total_CDR3 = sum(PID.count), 
            richness = n()/sum(PID.count),
            average_count = total_CDR3/richness,
            diversity = entropy(PID.fraction),
            simpsons = simpsons_index(PID.fraction))

TCRstats <- merge(TCRstats, metadata, by="exp")

# Write out stats, final and clean data to csvs for further analysis
write.csv(TCRstats, "summary_stats.csv", row.names = F) 
write.csv(exp_clean, "TCRdata_clean.csv", row.names = F)
write.csv(exp_final, "TCRdata_final.csv", row.names = F)

########
# Diversity evenness 50 (DE50) calculation
#######
# DE50 code only works in the below order
# Group by mouse, lists PID count in descending order
exp_group <- exp_final %>%
  as_tibble() %>%
  arrange(desc(PID.count)) %>%
  group_split(exp)

exp_group[[1]]$PID.count

#calculate DE50
DE50 <- list()
for (j in 1:length(exp_group)) {
  i = 1
  while (sum(exp_group[[j]]$PID.count[1:i]) < sum((exp_group[[j]]$PID.count)/2)) {
    i= i + 1 
  }
  DE50[[j]] <- i * 100 / nrow(exp_group[[j]])
}

DE50 <- do.call(rbind.data.frame, DE50)

write.csv(DE50, "DE50.csv") 
# Output is a list that you can put the put into 'summarystats.csv'
## The order of samples with DE50 calculations is the same order of samples in 'summary_stats.csv'.
### You can then copy and paste the DE50 values into the summary_stats.csv so all diversity metrics are together


