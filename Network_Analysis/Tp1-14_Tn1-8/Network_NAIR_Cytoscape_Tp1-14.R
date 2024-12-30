library(NAIR)
library(dplyr)
library(igraph)
library(ggplot2)

setwd("D:/Mitch/Documents/Honours/Project/Network_analysis/Tp1-14_Tn1-8/IGRP_Tp1-14_Tn1-8_data/TETpos")
TETpos_files <- list.files(pattern = "\\.csv$")
for (file in TETpos_files) {
  df_name <- gsub("\\.csv$", "", file)
  assign(df_name, read.csv(file, stringsAsFactors = FALSE))
}

setwd("D:/Mitch/Documents/Honours/Project/Network_analysis/Tp1-14_Tn1-8/IGRP_Tp1-14_Tn1-8_data/CD8")
CD8_files <- list.files(pattern = "\\.csv$")
for (file in CD8_files) {
  df_name <- gsub("\\.csv$", "", file)
  assign(df_name, read.csv(file, stringsAsFactors = FALSE))
}


setwd("D:/Mitch/Documents/Honours/Project/Network_analysis/Tp1-14_Tn1-8")


# Rename all TETpos and CD8 dataframes for all 8 mice to _ instead of -

CD8_1 <- `CD8-1`
CD8_2 <- `CD8-2`
CD8_3 <- `CD8-3`
CD8_4 <- `CD8-4`
CD8_5 <- `CD8-5`
CD8_6 <- `CD8-6`
CD8_7 <- `CD8-7`
CD8_8 <- `CD8-8`

TETpos_1 <- `TETpos-1`
TETpos_2 <- `TETpos-2`
TETpos_3 <- `TETpos-3`
TETpos_4 <- `TETpos-4`
TETpos_5 <- `TETpos-5`
TETpos_6 <- `TETpos-6`
TETpos_7 <- `TETpos-7`
TETpos_8 <- `TETpos-8`
TETpos_9 <- `TETpos-9`
TETpos_10 <- `TETpos-10`
TETpos_11 <- `TETpos-11`
TETpos_12 <- `TETpos-12`
TETpos_13 <- `TETpos-13`
TETpos_14 <- `TETpos-14`

# Remove the old dataframes with "-" in their names, create exp (all samples), and TETpos 1-8, TETpos 1-14, TETneg 1-8 files for later reference

rm(`CD8-1`, `CD8-2`, `CD8-3`, `CD8-4`, `CD8-5`, `CD8-6`, `CD8-7`, `CD8-8`)
rm(`TETpos-1`, `TETpos-2`, `TETpos-3`, `TETpos-4`, `TETpos-5`, `TETpos-6`, `TETpos-7`, `TETpos-8`, `TETpos-9`, `TETpos-10`, `TETpos-11`, `TETpos-12`, `TETpos-13`, `TETpos-14`)

exp <- rbind(CD8_1, CD8_2, CD8_3, CD8_4, CD8_5, CD8_6, CD8_7, CD8_8, TETpos_1, TETpos_2, TETpos_3, TETpos_4, TETpos_5, TETpos_6, TETpos_7, TETpos_8, TETpos_9, TETpos_10, TETpos_11, TETpos_12, TETpos_13, TETpos_14)
Tn1_8 <- rbind(CD8_1, CD8_2, CD8_3, CD8_4, CD8_5, CD8_6, CD8_7, CD8_8)
Tp1_8 <- rbind(TETpos_1, TETpos_2, TETpos_3, TETpos_4, TETpos_5, TETpos_6, TETpos_7, TETpos_8)
Tp1_14 <- rbind(TETpos_1, TETpos_2, TETpos_3, TETpos_4, TETpos_5, TETpos_6, TETpos_7, TETpos_8, TETpos_9, TETpos_10, TETpos_11, TETpos_12, TETpos_13, TETpos_14)
# Read data and metadata

metadata <- read.csv("IGRP_Tp1-14_Tn1-8_data/metadata.csv", header = TRUE)


# Merge clones that have the same aa seq but different nuc sequence for further graphs

net_merge <- exp %>%
  left_join(metadata) %>%
  group_by(exp, aaSeqCDR3, mouse, cells) %>% # add/change variables from metadata
  summarise(PID.count_sum = sum(PID.count),
            PID.fraction_sum = sum(PID.fraction),
            clone.fraction_sum = sum(cloneFraction),
            clone.count_sum = sum(cloneCount))

Tn1_8_merge_aa <- Tn1_8 %>%
  left_join(metadata) %>%
  group_by(exp, aaSeqCDR3, mouse, cells) %>% # add/change variables from metadata
  summarise(PID.count_sum = sum(PID.count),
            PID.fraction_sum = sum(PID.fraction),
            clone.fraction_sum = sum(cloneFraction),
            clone.count_sum = sum(cloneCount))

Tp1_8_merge_aa <- Tp1_8 %>%
  left_join(metadata) %>%
  group_by(exp, aaSeqCDR3, mouse, cells) %>% # add/change variables from metadata
  summarise(PID.count_sum = sum(PID.count),
            PID.fraction_sum = sum(PID.fraction),
            clone.fraction_sum = sum(cloneFraction),
            clone.count_sum = sum(cloneCount))

Tp1_14_merge_aa <- Tp1_14 %>%
  left_join(metadata) %>%
  group_by(exp, aaSeqCDR3, mouse, cells) %>% # add/change variables from metadata
  summarise(PID.count_sum = sum(PID.count),
            PID.fraction_sum = sum(PID.fraction),
            clone.fraction_sum = sum(cloneFraction),
            clone.count_sum = sum(cloneCount))


setwd("D:/Mitch/Documents/Honours/Project/Processing/Data/Grouped_merged_data")

write.csv(Tn1_8, "Tn1_8.csv", row.names = FALSE)
write.csv(Tp1_8, "Tp1_8.csv", row.names = FALSE)
write.csv(Tp1_14, "Tp1_14.csv", row.names = FALSE)
write.csv(Tn1_8_merge_aa, "Tn1_8_merge_aa.csv", row.names = FALSE)
write.csv(Tp1_8_merge_aa, "Tp1_8_merge_aa.csv", row.names = FALSE)
write.csv(Tp1_14_merge_aa, "Tp1_14_merge_aa.csv", row.names = FALSE)

setwd("D:/Mitch/Documents/Honours/Project/Network_analysis/Tp1-14_Tn1-8")

net_merge <- net_merge %>%
  mutate_if(is.factor, ~as.numeric(as.character(.)))


# Filter data for TETpos and top 50, 100, 200 etc sequences

net_merge_TETpos <- net_merge %>%
  filter(cells %in% c('TETpos'))

final_network_TETpos_75 <- net_merge_TETpos %>%
  group_by(exp) %>%
  arrange(desc(PID.count_sum))%>%
  mutate(rank = 1:length(exp)) %>%
  filter(rank <= 75) %>% 
  group_by(exp, aaSeqCDR3, mouse) %>%
  group_by(aaSeqCDR3)


# Generate count cvs for use in cytoscape visualisation, keep only AaSeqCDR3 and count columns, remove duplicates so each sequence and its count appear once

final_network_TETpos_75_counts <- final_network_TETpos_75 %>% 
  group_by(aaSeqCDR3) %>%
  mutate(unique_mouse_count = n_distinct(mouse)) %>% 
  ungroup()

keep_column <- c("aaSeqCDR3", "unique_mouse_count")

final_network_TETpos_75_counts <- final_network_TETpos_75_counts %>% 
  select(all_of(keep_column))

final_network_TETpos_75_counts <- final_network_TETpos_75_counts %>%
  distinct(aaSeqCDR3, .keep_all = TRUE)

write.csv(final_network_TETpos_75_counts, "TETpos_1-14_75_counts.csv", row.names = FALSE, quote = FALSE)
  

# merge sequences that appear across multiple mice (if this isn't done, many duplicate networks form, we have mice_count data already in the counts file for use as node size parameter)

final_network_TETpos_75_seq <- final_network_TETpos_75 %>% 
  select(aaSeqCDR3)

final_network_TETpos_75_seq <- final_network_TETpos_75_seq %>% 
  distinct(aaSeqCDR3)

# Generate TETpos network and node and cluster analysis output files

net <- buildNet(final_network_TETpos_75_seq,
                seq_col = "aaSeqCDR3",
                dist_type = "lev",
                node_stats = TRUE,
                stats_to_include = "all",
                cluster_stats = TRUE,
                color_scheme = c("default"),
                node_size_limits = c(2.0, 2.5),
                plot_title = c("TETpos Top 75"),
                plot_subtitle = NULL,
                print_plots = TRUE,
                output_dir = ("TETpos_75_outputs"),
                output_type = "individual"
)

net <- labelNodes(net, "aaSeqCDR3", size = 0.5)

net$plots[[1]]

# Outputs of Network analysis are saved to outputs folder, some of these required for visualisation in cytoscape.




# Now modify output data for visualisation in Cytoscape.


# Import the Node metadata.csv and network edge list.txt files.

node_seq <- read.csv("TETpos_75_outputs/MyRepSeqNetwork_NodeMetadata.csv", header = TRUE, stringsAsFactors = FALSE)

edge_list <- read.table("TETpos_75_outputs/MyRepSeqNetwork_EdgeList.txt", sep = "", header = FALSE)


# Add ascending numeric count column starting at 0 to the NodeMetaData (this will match perfectly with the number designation assigned to each individual AAseqCDR3 in the edge_list output from NAIR)

new_column <- 0:(nrow(node_seq) - 1)

node_seq <- cbind(new_column, node_seq)


# remove all columns apart from ascending numeric count column and aaSeqCDR3, the number in the ascending numeric column represents both the numbers in the edge_list columns and the amino acid in the same row of the aaSeqCDR3 column.

node_seq <- node_seq %>% 
  select(new_column, aaSeqCDR3)


# Merge edge_list with node_seq in new dataframe to replace edge_list numbers (these represent connected nodes) with amino acid sequences as assigned by NAIR

network_table <- edge_list %>%
  left_join(node_seq, by = c("V1" = "new_column")) %>%
  rename(AA_Seq1 = aaSeqCDR3) %>%
  select(-V1) %>%
  left_join(node_seq, by = c("V2" = "new_column")) %>%
  rename(AA_Seq2 = aaSeqCDR3) %>%
  select(AA_Seq1, AA_Seq2)


# Write to csv and import into Cytoscape with "file > import > network from file" or drag and drop into network file frame. 

write.csv(network_table, "Cytoscape_inputs/Cytoscape_TETpos_1-14_75.csv", row.names = FALSE, quote = FALSE)



# -------- REPEAT PROCESS FOR TETneg --------


# Filter data for TETneg and top 50, 100, 200 etc sequences

net_merge_TETneg <- net_merge %>%
  filter(cells %in% c('CD8'))

final_network_TETneg_75 <- net_merge_TETneg %>%
  group_by(exp) %>%
  arrange(desc(PID.count_sum))%>%
  mutate(rank = 1:length(exp)) %>%
  filter(rank <= 75) %>% 
  group_by(exp, aaSeqCDR3, mouse) %>%
  group_by(aaSeqCDR3)

# Generate count csv for use in cytoscape visualisation, keep only AaSeqCDR3 and count columns, remove duplicates so each sequence and its count appear once

final_network_TETneg_75_counts <- final_network_TETneg_75 %>% 
  group_by(aaSeqCDR3) %>%
  mutate(unique_mouse_count = n_distinct(mouse)) %>% 
  ungroup()

keep_column <- c("aaSeqCDR3", "unique_mouse_count")

final_network_TETneg_75_counts <- final_network_TETneg_75_counts %>% 
  select(all_of(keep_column))

final_network_TETneg_75_counts <- final_network_TETneg_75_counts %>%
  distinct(aaSeqCDR3, .keep_all = TRUE)

write.csv(final_network_TETneg_75_counts, "TETneg_75_outputs/TETneg_75_counts.csv", row.names = FALSE, quote = FALSE)


# merge sequences that appear across multiple mice.

final_network_TETneg_75_seq <- final_network_TETneg_75 %>% 
  select(aaSeqCDR3)

final_network_TETneg_75_seq <- final_network_TETneg_75_seq %>% 
  distinct(aaSeqCDR3)


# Generate TETneg network and node and cluster analysis output files

net1 <- buildNet(final_network_TETneg_75_seq,
                 seq_col = "aaSeqCDR3",
                 dist_type = "lev",
                 node_stats = TRUE,
                 stats_to_include = "all",
                 cluster_stats = TRUE,
                 color_scheme = c("default"),
                 node_size_limits = c(2.0, 2.5),
                 plot_title = c("TETneg Top 75"),
                 plot_subtitle = NULL,
                 print_plots = TRUE,
                 output_dir = ("TETneg_75_outputs"),
                 output_type = "individual"
)

net1 <- labelNodes(net1, "aaSeqCDR3", size = 0.5)

net1$plots[[1]]


# Outputs of Network analysis are saved to outputs folder.

# Import the Node metadata.csv and network edge list.txt files.

node_seq1 <- read.csv("TETneg_75_outputs/MyRepSeqNetwork_NodeMetadata.csv", header = TRUE, stringsAsFactors = FALSE)

edge_list1 <- read.table("TETneg_75_outputs/MyRepSeqNetwork_EdgeList.txt", sep = "", header = FALSE)


# Add ascending numeric count column starting at 0 to the NodeMetaData

new_column1 <- 0:(nrow(node_seq1) - 1)

node_seq1 <- cbind(new_column1, node_seq1)


# remove all columns apart from ascending numeric count column and aaSeqCDR3, the number in the ascending numeric column represents both the numbers in the edge_list columns and the amino acid in the same row of the aaSeqCDR3 column.

node_seq1 <- node_seq1 %>% 
  select(new_column1, aaSeqCDR3)


# Merge edge_list with node_seq in new dataframe to replace edge_list numbers (these represent connected nodes) with amino acid sequences as assigned by NAIR

network_table1 <- edge_list1 %>%
  left_join(node_seq1, by = c("V1" = "new_column1")) %>%
  rename(AA_Seq1 = aaSeqCDR3) %>%
  select(-V1) %>%
  left_join(node_seq1, by = c("V2" = "new_column1")) %>%
  rename(AA_Seq2 = aaSeqCDR3) %>%
  select(AA_Seq1, AA_Seq2)


# Write to csv and import into Cytoscape with "file > import > network from file" or drag and drop into network file frame. 

write.csv(network_table1, "Cytoscape_inputs/Cytoscape_TETneg_75.csv", row.names = FALSE, quote = FALSE)


# For count csv files, first import associated network file, then add count file through "file > import > table from file", set target table data to "to selected networks only" and select network from list below, import data as "node table columns" and key column for networks to "shared name"