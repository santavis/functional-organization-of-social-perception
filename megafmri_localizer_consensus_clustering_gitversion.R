# Consensus clustering analysis of social perceptual features
#
# Severi Santavirta 07.03.2023

library(corrplot)
library(ConsensusClusterPlus)

## Load data
features = read.csv("/path/social_features.csv",sep = ";"); # Read average social feature time series for reliably evaluated features (45 features)

## Concensus clustering
d <- as.matrix(features)
title <- "hc_average_pearson"
cc <- ConsensusClusterPlus(d,maxK = 20,reps = 5000,pItem = 0.8,pFeature = 0.8,clusterAlg = "hc",distance = "pearson",plot = "png",title = title, innerLinkage = "average")
icl <- calcICL(cc,plot = "png",title = title)

# Plot the hierarchically ordered correlation matrix with k=13
corrplot(cor(features_45),tl.col = "black",method = "square",col=colorRampPalette(c("#2166AC","#4393C3","#92C5DE","#D1E5F0","#FDDBC7","#F4A582","#D6604D","#B2182B"))(20),order = "hclust",hclust.method = "average",addrect = 13)



