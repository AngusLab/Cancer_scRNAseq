BiocManager::install("edgeR")


library(edgeR)
#library(anndataR)
library(dplyr)
library(limma)
library(pheatmap)
library(RColorBrewer)

dir.create("EdgeR_DEA")
OutPath<-("EdgeR_DEA/")

counts<- read.csv("no_pseudo_replciates_counts.csv")
metadata<- read.csv("metadata_nopseudo_2_13.csv")
metadata$orig_sample=factor(metadata$org_sample)
metadata$group <- paste0(metadata$org_sample, ".", metadata$cluster)


# Making sample the rownames
rownames(metadata)<- metadata$pseudo
head(metadata)
(dim(metadata))
(colnames(metadata))
dim(counts)
#Making the row names of the meta equal the column name of the count dataframe
rownames(counts)<- counts$X; counts<- counts[,-c(1)]
metadata<- metadata[colnames(counts), , drop = FALSE]
dim(metadata)

# DEA by entire dataset MPNST vs PNF
#Making design
sample_names <- sub("_cl.*", "", colnames(counts))
counts_sample <- sapply(unique(sample_names), function(s)
  rowSums(counts[, sample_names == s, drop=FALSE]))

counts_sample <- as.matrix(counts_sample)
metadata_sample<- metadata[!duplicated(metadata$orig_sample),]

group<- factor(metadata_sample$condition)
replicate<- factor(metadata_sample$org_sample)
design <- model.matrix(~ 0+group) 
design  
#Setting up list

y<- DGEList(counts = counts_sample, group = group)
y
keep<- filterByExpr(y, min.count=10, min.total.count = 20)
table(keep)
y <- y[keep, , keep.lib.sizes=FALSE]
y<- calcNormFactors(y)
lcpm <- cpm(y, log=TRUE, prior.count=2)
write.csv(lcpm, file=paste0(OutPath,"logCPM MPNST vs PNF.csv"))
x<- as.factor(y$samples$group)

png(filename = paste0(OutPath, "MDS altogether.png"),
    width = 2000, height = 1500, res = 150, type = "cairo")
plotMDS(y, pch=16, col=c(2:8)[x], main="MDS")
legend("bottomright", legend=paste0("sample", levels(x)),
pch=16, col=2:8, cex=0.9)
dev.off()
y <- estimateDisp(y, design = design)
fit <- glmQLFit(y, design)
list("fit"=fit, "design"=design, "y"=y)
cont <- makeContrasts(groupMPNST - groupPNF, levels = design)
qlf <- glmQLFTest(fit, contrast = cont)
tt <- topTags(qlf, n = Inf)
result <- tt$table
df<- as.data.frame(result)
png(filename = paste0(OutPath, "DEA altogether.png"),
    width = 2000, height = 1500, res = 150, type = "cairo")
plotSmear(qlf, de.tags = rownames(tt)[which(tt$FDR<0.1)])
dev.off()
write.csv(df,file=paste0(OutPath,"DEA_MPNST_vs_PNF_altogether.csv"))
print("finished comparing MPNST vs other tumor types")

#Grabbing ZNF423 and plotting a bar plot

lcpm2<- as.data.frame(t(lcpm))
lcpm2<- as.data.frame(lcpm2$ZNF423); colnames(lcpm2)<-"ZNF423"
lcpm2$Sample<- colnames(lcpm)

meta_ZNF423<- metadata_sample%>%select(org_sample, condition)
lcpm2<- inner_join(lcpm2, meta_ZNF423, by = c("Sample"="org_sample"))
df$Gene<- rownames(df)
df_ZNF423<- df%>%filter(Gene=="ZNF423")
pval<- df_ZNF423$PValue[1]
fdr<- df_ZNF423$FDR[1]

p_to_stars <- function(p) {
  if (p < 1e-4) return("****")
  else if (p < 1e-3) return("***")
  else if (p < 1e-2) return("**")
  else if (p < 0.05) return("*")
  else return("ns")
}
stars_p  <- p_to_stars(pval)
stars_fdr <- p_to_stars(fdr)

p_text <- paste0("edgeR p = ", formatC(pval, format = "g", digits = 3)," ",stars_p,
                 "\nFDR = ", formatC(fdr, format = "g", digits = 3)," ",stars_fdr)
y_max <- max(lcpm2$ZNF423, na.rm = TRUE)
y_pos <- y_max + 0.08 * (y_max - min(lcpm2$ZNF423, na.rm = TRUE))

library(ggpubr)

a<-ggboxplot(lcpm2,"condition", "ZNF423", 
          fill = "condition", 
          palette = c("orange","blue"),
          title = "ZNF423 expression in PNF vs. MPNST",
          subtitle="Pseudobulked scRNA-sequencing",
          xlab= NULL,
          ylab= "ZNF423 log2CPM",
          legend="none",
          add = "jitter")+
  font("title", size =15)+
  theme_bw()+
  theme(legend.position = "none",
        axis.title.x = element_blank())+
  annotate("text", x = 1.5, y = y_pos, label = p_text, hjust = 0.5, size = 4)

ggsave("ZNF423 expression in MPNST.png", a, device = "png", width = 6, height = 6)