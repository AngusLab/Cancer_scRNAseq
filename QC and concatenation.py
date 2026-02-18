import scanpy as sc
import anndata as ad
import pandas as pd
import os
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from scipy.stats import median_abs_deviation
from contextlib import redirect_stdout
import logging
from pathlib import Path


def setup_logger(out_prefix, log="log"):
    Path(out_prefix).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(out_prefix, f"{log}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler() 
        ]
    )


def initial_qc (file_name='_filtered_feature_bc_matrix',
                outdir="Path/to/outdir",
                log="initial_qc"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    setup_logger(outdir, log = log)
    sc.settings.figdir = os.path.abspath(outdir)

    #Initialise empty index
    adatas={}
    # Get all CSV files in the current directory
    h5_files = [f for f in os.listdir() if f.endswith('.h5')]

    for file in h5_files:
        # Extract the sample name
        base_name = file.replace(file_name, '').replace(".h5",'')
        logging.info(base_name)
        # Read the h5 file and add the sample name
        adata_new = sc.read_10x_h5(file)
        adata_new.var_names_make_unique()
        adata_new.obs_names_make_unique()
        adata_new.obs['sample'] = base_name
        logging.info (adata_new.var.head())
        adatas[base_name] = adata_new


    ## QCing for loop/function


    #Getting rid of things based on nmad
    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier
    
    def perform_qc(adatas, outdir):
        os.makedirs(outdir, exist_ok=True)
        for sample_name, adata in adatas.items():
            logging.info(f"QC for {sample_name}")
            #Create PDF and log file
            pdf_path = f"{outdir}/{sample_name}_QC_output.pdf"
            log_path = f"{outdir}/{sample_name}_QC_log.txt"
            pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

            with open(log_path, "w") as log_file:
                with redirect_stdout(log_file):
                    logging.info(f"QC for {sample_name}")
                    
                    # mitochondrial genes, "MT-" for human, "Mt-" for mouse. 
                    adata.var["mt"] = adata.var_names.str.startswith("MT-")
                    #ribosomal genes, "RPS","RPL" for human, "Rps", "Rpl" for mouse
                    adata.var["ribo"]=adata.var_names.str.startswith(("RPS", "RPL"))
                    #hemoglobin genes, "HB..P" for human, "Hb...p" for mouse
                    adata.var["hb"]=adata.var_names.str.startswith("^HB[^(P)]")
                    logging.info(adata.var.mt.value_counts())
                    logging.info(adata.var.ribo.value_counts())
                    logging.info(adata.var.hb.value_counts())
                    #Calculates QC metrics of counts for these populations
                    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)
                    logging.info(adata)
                    p2=sns.displot(adata.obs["total_counts"], bins=100, kde=False).figure
                    plt.title( f"{sample_name} before removing outliers")
                    pdf.savefig(p2)
                    plt.close(p2)
                    p3= sc.pl.violin(adata, "pct_counts_mt",show =False)
                    p3 = plt.gcf()
                    pdf.savefig(p3)
                    plt.close(p3)
                    p4= sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",show =False)
                    p4 = plt.gcf()
                    pdf.savefig(p4)
                    plt.close(p4)

                    #Removing outliers via mnad
                    adata.obs["outlier"] = (
                        is_outlier(adata, "log1p_total_counts", 5)
                        | is_outlier(adata, "log1p_n_genes_by_counts", 5)
                        | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
                    )
                    logging.info(adata.obs.outlier.value_counts())

                    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
                        adata.obs["pct_counts_mt"] > 8
                    )
                    logging.info(adata.obs.mt_outlier.value_counts())

                    logging.info(f"Total number of cells: {adata.n_obs}")
                    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

                    logging.info(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

                    p5 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt",show =False)
                    p5 = plt.gcf()
                    plt.title( f"{sample_name} after removing outliers")
                    pdf.savefig(p5)
                    plt.close(p5)

                    pdf.close()
                    #Removing doublets
                    sc.pp.scrublet(adata, batch_key="sample")
                    
                    logging.info(adata.obs.predicted_doublet.value_counts()) #2064 cells are predicted to be doublets
                    logging.info(adata.obs.doublet_score.value_counts())

                    #saving adata
                    adata.write_h5ad(f"{sample_name} QC and defined doublets.h5ad")
                    logging.info(f"Finished QC for {sample_name}")


    perform_qc(adatas, "QC_for_each_sample")

    #After QCing each sample
    import os
    #import pandas as pd
    #Initialise empty index
    adatas={}
    # Get all CSV files in the current directory
    h5ad_files = [f for f in os.listdir() if f.endswith('QC and defined doublets.h5ad')]

    for file in h5ad_files:
        # Extract the sample name
        base_name = file.replace(' QC and defined doublets.h5ad', '')
        logging.info(base_name)
        # Read the h5 file and add the sample name
        adata_new = sc.read_h5ad(file)
        adata_new.var_names_make_unique()
        adata_new.obs_names_make_unique()
        adata_new.obs['sample'] = base_name
        #Get rid of doublets
        
        #print (adata_new.var.head())
        adatas[base_name] = adata_new
        

    adata=ad.concat(adatas, label="sample", join = "outer")

    logging.info(adata.obs["sample"].value_counts())

    adata.obs_names_make_unique()
    logging.info(adata.obs.predicted_doublet.value_counts())

    adata.layers["counts"] = adata.X.copy() #saving counts data
    adata.layers

    #Checking to make sure adata.layers["counts"] is still the raw. It should match the adata.X layer
    logging.info("Counts layer stats:")
    logging.info("Min:", adata.layers["counts"].min())
    logging.info("Max:", adata.layers["counts"].max())
    logging.info("Mean:", adata.layers["counts"].mean())

    logging.info("X layer stats:")
    logging.info("Min:", adata.X.min())
    logging.info("Max:", adata.X.max())
    logging.info("Mean:", adata.X.mean())


    from matplotlib import pyplot as plt
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    # log1p transform
    adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)


    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
    axes[0].set_title("Total counts")
    p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1])
    axes[1].set_title("Shifted logarithm")
    plt.show()


    adata.write_h5ad("Noramlized counts of concatenated data.h5ad")
    return adata



def qc_and_hvg_subset(metatdata_file='metadata.csv',
                outdir="Path/to/outdir",
                log="final_qc_and_hvg"):

    Path(outdir).mkdir(parents=True, exist_ok=True)
    setup_logger(outdir, log = log)
    sc.settings.figdir = os.path.abspath(outdir)

    #Initialise empty index
    adatas={}
    # Get all CSV files in the current directory
    h5ad_files = [f for f in os.listdir() if f.endswith('QC and defined doublets.h5ad')]

    for file in h5ad_files:
        # Extract the sample name
        base_name = file.replace(' QC and defined doublets.h5ad', '')
        logging.info(base_name)
        # Read the h5 file and add the sample name
        adata_new = sc.read_h5ad(file)
        adata_new.var_names_make_unique()
        adata_new.obs_names_make_unique()
        adata_new = adata_new[~adata_new.obs['predicted_doublet'], :].copy()
        adata_new.obs['sample'] = base_name
        adatas[base_name] = adata_new
        

    adata=ad.concat(adatas, label="sample", join = "outer")

    logging.info(adata.obs["sample"].value_counts())

    adata.obs_names_make_unique()
    logging.info(adata.obs.predicted_doublet.value_counts())

    #Reading in metatdata
    df = pd.read_csv(metatdata_file)
    logging.info(df.head)

    metadata_dict = df.set_index('sample').to_dict('index')

    adata.obs['tumorID'] = None
    adata.obs['tumor_type'] = None
    adata.obs['batch']= None

    # Iterate through the rows of adata.obs
    for index, row in adata.obs.iterrows():
        sample_id = row['sample']

        # Check if the sample exists in the metadata dictionary
        if sample_id in metadata_dict:
            # Get the corresponding metadata
            sample_metadata = metadata_dict[sample_id]

            # Assign the tumorID and tumor_type to the adata.obs row
            adata.obs.loc[index, 'tumorID'] = sample_metadata.get('tumorID')
            adata.obs.loc[index, 'tumor_type'] = sample_metadata.get('tumor_type')
            adata.obs.loc[index, 'batch'] = sample_metadata.get('batch')

    # Verify the update
    logging.info(adata.obs[['sample', 'tumorID', 'tumor_type','batch']].head())
    #saving counts data
    adata.layers["counts"] = adata.X.copy() 
    #Normalizing
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    # log1p transform
    adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)
    adata.obs_names_make_unique()
    #Saving here so if it crashes, we start from this adata frame
    adata.write_h5ad("adata_right_before_PRHVG.h5ad")
    logging.info(adata.shape)
    #Defining HVGs
    sc.experimental.pp.recipe_pearson_residuals(adata,
                                                    batch_key = "sample",
                                                    n_top_genes= 10000)

    logging.info(adata.var.loc["ZNF423", "highly_variable"])

    adata = adata[:, adata.var["highly_variable"]].copy()
    logging.info(adata.shape)
    logging.info("subsetted highly variable genes")
    logging.info("ZNF423" in adata.var_names)
    #tsne
    sc.tl.tsne(adata, use_rep="X_pca")
    #umap
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    adata.write_h5ad("adata_HVG_pearson_residuals_no_doublets.h5ad")
    logging.info("Successfully subsetted")
    logging.info(adata)
    return adata
