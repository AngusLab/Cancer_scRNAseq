import logging
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import anndata as ad
import pandas as pd
import os
import tempfile
import scvi
import torch
import tensorflow as tf
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

def scvi_batch_correction(adata_path="adata_HVG_pearson_residuals_no_doublets.h5ad",
                          outdir= "/Path/to/outdir",
                          log="scvi_batch",
                          model_save="/Path/to/where/model/is/saved"):


    Path(outdir).mkdir(parents=True, exist_ok=True)
    setup_logger(outdir, log = log)
    sc.settings.figdir = os.path.abspath(outdir)

    sc.settings.verbosity = 0
    sc.settings.set_figure_params(
        dpi=80,
        facecolor="white",
        frameon=False,
    )
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("CUDA not available. Falling back to CPU.")
    #List all physical devices (CPU and GPU)
    physical_devices = tf.config.list_physical_devices()
    logging.info("Physical devices:", physical_devices)

    #List only GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info("Found GPUs:", gpus)
    else:
        logging.info("No GPUs found.")

    logging.info("TensorFlow version:", tf.__version__)
    logging.info("GPUs available:", tf.config.list_physical_devices('GPU'))
    logging.info("CUDA enabled:", tf.test.is_built_with_cuda())
    logging.info("GPU name:", tf.test.gpu_device_name())
    #Reading in adata
    adata= sc.read_h5ad(adata_path)
    logging.info(adata)
    logging.info("adata.shape:",adata.shape)
    logging.info("counts layer shape:", getattr(adata.layers.get("counts"), "shape", None))
    logging.info(f"obs keys: {list(adata.obs_keys())}")


    counts_layer = "counts" if "counts" in adata.layers else None
    if counts_layer is None:
        logging.info("No 'counts' layer found; scvi will use adata.X. If you have a 'counts' layer, rename it to 'counts'.")

    logging.info(adata)

    sc.pl.umap(adata,color=["total_counts", "sample"], 
            save = "_with pearson residuals without doublets.png",
            show = False)
    sc.pl.umap(
        adata,
        color=["total_counts", "pct_counts_mt", "predicted_doublet", "doublet_score"],
        save ="_with pearson residuals of QC scores without doublets.png",
        ncols=2,
        show = False
    )
    sc.pl.tsne(adata, color=["total_counts", "sample"],
            ncols = 1,
            save = "_with experimental residuals without doublets.png", 
            show = False)

    # Choose continuous covariates
    covariates = ["total_counts", "pct_counts_mt", "doublet_score"]

    # Extract first few PCs
    pcs = pd.DataFrame(adata.obsm["X_pca"][:, :5], columns=[f"PC{i+1}" for i in range(5)])

    # Add covariates
    for cov in covariates:
        pcs[cov] = adata.obs[cov].values

    # Correlation heatmap
    corr = pcs.corr().loc[covariates, [f"PC{i+1}" for i in range(5)]]
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between Covariates and Principal Components")
    plt.savefig('Correlation between Covariates and Principal Components.png')
    plt.show()

    scvi.model.SCVI.setup_anndata(adata, layer="counts",batch_key="sample",continuous_covariate_keys=["pct_counts_mt"])
    model = scvi.model.SCVI(adata)
    logging.info(model)
    model.train()
    model.save(model_save, overwrite=True)
    logging.info(f"Saved SCVI model")

    adata= sc.read_h5ad("adata_HVG_pearson_residuals_no_doublets.h5ad")

    model = scvi.model.SCVI.load(model_save, adata=adata)
    SCVI_LATENT_KEY = "X_scVI"

    latent = model.get_latent_representation()
    adata.obsm[SCVI_LATENT_KEY] = latent
    logging.info(f"Latent shape: {latent.shape}")

    SCVI_NORMALIZED_KEY = "scvi_normalized"

    adata.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(library_size=10e4)
    logging.info("Generating pca, tsnie, etc.")
    # re generate PCA then generate UMAP plots
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=["sample"],
        ncols=4,
        frameon=False,
        save="_after batch effect using latent space without doublets_take 2.png",
        show=False
    )
    adata.write_h5ad("modeled_adata_afterumap without doublets.h5ad", compression="gzip")
    logging.info("Finished and saved modeled adata and figures.")
    return adata


def leiden_clustering(adata_path="modeled_adata_afterumap without doublets.h5ad",
                      outdir= "/Path/to/outdir",
                      log="leiden_clustering"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    setup_logger(outdir, log = log)
    sc.settings.figdir = os.path.abspath(outdir)
    adata = sc.read_h5ad(adata_path)
    # neighbors were already computed using scVI
    SCVI_CLUSTERS_KEY = "leiden_scVI"
    sc.tl.leiden(adata, key_added=SCVI_CLUSTERS_KEY, resolution=1.0)
    logging.info(adata.obs[SCVI_CLUSTERS_KEY].value_counts())
    logging.info(adata.obs)

    sc.pl.umap(adata,
            color="sample",
            size =2,
            save="Samples after batch correction.png",
            show = False
    )
    sc.pl.umap(
        adata,
        color=['leiden_scVI'],
        frameon=False,
        save="_leiden_scVI.png",
        show = False
    )

    #Re-assess quality control and cell filtering
    sc.pl.umap(
        adata,
        color = ["leiden_scVI", "predicted_doublet", "doublet_score"],
        #increase horizontal space between panels
        wspace=0.5,
        size = 3,
        frameon=True,
        save = "_leiden clustering with doublet prediction without doublets.png",
        show = False
    )

    sc.pl.umap(
        adata,
        color=["leiden_scVI", "log1p_total_counts", "pct_counts_ribo", "log1p_n_genes_by_counts"],
        wspace=0.5,
        ncols=2,
        frameon=True,
        save = "_leiden clustering with QC stats without doublets.png",
        show = False
    )
    logging.info("Wrote Umaps")
    # Manually annotating cluster
    for res in [0.02,0.05,0.1,0.25,0.5, 1.0,1.25, 1.5, 1.75,2.0]:
        sc.tl.leiden(
            adata, key_added=f"leiden_res_scvi_{res:4.2f}", resolution=res, flavor="igraph"
        )
    logging.info("Did leiden calculations")
    sc.pl.umap(
        adata,
        color=["leiden_res_scvi_0.02","leiden_res_scvi_0.05","leiden_res_scvi_0.10", "leiden_res_scvi_0.50","leiden_res_scvi_1.00","leiden_res_scvi_1.25","leiden_res_scvi_1.50","leiden_res_scvi_1.75","leiden_res_scvi_2.00"],
        legend_loc="on data",
        ncols=3,
        frameon=True,
        save="Different leiden resolutions without doublets.png"
    )   

    adata.write_h5ad('adata_Leiden_clustering without doublets.h5ad', compression="gzip")
    logging.info("Wrote adata object")
    return adata

def visualize_leiden_cluster_of_choice(adata_path="adata_Leiden_clustering without doublets.h5ad",
                                       resolution="leiden_res_scvi_1.00"):
        
        adata=sc.read_h5ad(adata_path)

        sc.pl.umap(
        adata,
        color = [resolution],
        legend_loc="on data",
        show = False,
        save = "_leiden_on_data.png")


