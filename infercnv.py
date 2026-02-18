
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import infercnvpy as cnv
import os, tempfile
import traceback

tempfile.tempdir = "/Path/to/Temp"
adata = sc.read_h5ad("/Path/to/leiden/clustered/data")

print("n genes:", adata.n_vars)
print("example var_names:", list(adata.var_names[:30]))
print("columns in adata.var:", list(adata.var.columns))



gtf_path='/Path/to/gencode.v38.annotation.gtf'
try:
    cnv.io.genomic_position_from_gtf(gtf_path, adata=adata, gtf_gene_id='gene_name', inplace=True)
except Exception:
    print("Error", Exception)
    traceback.print_exec()

annotated=adata.var["chromosome"].notna().sum()
missing_mask=~adata.var["chromosome"].notna()
missing=adata.var_names[missing_mask]

#Check how many have been annotated
print(adata.var['chromosome'].unique())
print("annoted after gene_name match:", annotated, "/", adata.n_vars)
print("missing count:", missing.shape[0])
print("examples missing", list(missing[:30]))
print("Annotated after gene_name (parsed GTF) join:", adata.var['chromosome'].notna().sum(), "/", adata.n_vars)


def cnv_leiden(adata,
               reference_key="leiden_res_scvi_1.00",
               reference_cat=["6","36","22","7","25", "12","15","8","34","35","40"]):
    
    # Assume adata.var has columns: chromosome, start, end (after GTF mapping attempt)
    mask_complete = adata.var[['chromosome', 'start', 'end']].notna().all(axis=1)
    adata = adata[:, mask_complete].copy()
    print("adata shape:", adata.shape)
    print("about to start infercnv")

    cnv.tl.infercnv(adata, reference_key=reference_key, reference_cat=reference_cat, 
                    window_size=100, step=10,chunksize=1000 ,exclude_chromosomes=['chrX', 'chrY', 'chrM'])

    #cnv.pl.chromosome_heatmap(adata, groupby="leiden_res_scvi_1.00", dendrogram=True, show=False, save="_cnv.png")

    #Making UMAP
    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    cnv.tl.leiden(adata)
    cnv.tl.umap(adata)
    cnv.tl.cnv_score(adata)
    cnv_score=adata.obs.groupby(reference_key)["cnv_score"].mean()
    print(cnv_score)
    cnv_score.to_csv("grouped_cnv_scores.csv")
    adata.write_h5ad("adata_with_cnv.h5ad",compression= "gzip")
    cnv.pl.umap(adata, color="cnv_score", show = False, save = "_cnv_umap.png")

    sc.pl.umap(
        adata,
        color=["cnv_score",reference_key],
        legend_loc="on data",
        ncols=2,
        show = False,
        frameon=False,
        save=f"_CNV_scores_{reference_key}.png"
    )

    print(adata.var.head())
    adata.obs["malignant_type"]=None

    for i in adata.obs.index:
        if adata.obs.loc[i,"cnv_score"] >0.04:
            adata.obs.loc[i,"malignant_type"]="tumor"
        elif adata.obs.loc[i,"cnv_score"] <0.02:
            adata.obs.loc[i,"malignant_type"]="normal"
        else:
            adata.obs.loc[i,"malignant_type"]="borderline"

    print(adata.obs["malignant_type"].head())
    percent=adata.obs["malignant_type"].value_counts(normalize=True).rename_axis("malignant_type").reset_index(name= "proportion")
    print(percent)


    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    results=[]
    pdf = "histogram_of_cnv_scores.pdf"
    cluster = adata.obs[reference_key].unique()
    print(cluster)
    with PdfPages(pdf) as pdf:
        for c in cluster:
            Xsub = adata[adata.obs[reference_key]==c]
            #print(Xsub.obs["leiden_res_scvi_1.00"].head())
            percent=Xsub.obs["malignant_type"].value_counts(normalize=True).rename_axis("malignant_type").reset_index(name= "proportion")
            #print(percent)
            try:
                normal = percent.loc[percent["malignant_type"]=="normal", "proportion"].values[0]
                if normal is not None:
                    print(normal)
            except Exception as e:
                print(f"Found exception:{e} in cluster {c} for normal")
            try:
                borderline = percent.loc[percent["malignant_type"]=="borderline", "proportion"].values[0]
                if borderline is not None:
                    print(borderline)
            except Exception as e:
                print(f"Found exception:{e} in cluster {c} for borderline")
            try:
                tumor = percent.loc[percent["malignant_type"]=="tumor", "proportion"].values[0]
                if tumor is not None:
                    print(tumor)
            except Exception as e:
                print(f"Found exception:{e} in cluster {c} for tumor")
            mean=Xsub.obs["cnv_score"].mean()
            print("The mean is ",mean)
            median=Xsub.obs["cnv_score"].median()
            print("The median is ",median)     
            results.append({
                    "normal":normal,
                    "borderline":borderline,
                    "tumor":tumor,
                    "cluster": c,
                    "mean": mean,
                    "median": median
                })
            fil=plt.figure()
            plt.hist(Xsub.obs["cnv_score"], bins = 30, color= "lavender", edgecolor="black")
            plt.xlabel("CNV Scores")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of CNV scores for cluster {c}")
            pdf.savefig(fil)
            plt.close(fil)

    panel_df = pd.DataFrame(results)
    panel_df.to_csv("Summary of CNV_scores.csv", index=False)
    return adata
