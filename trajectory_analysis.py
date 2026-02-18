import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse as sp
import seaborn as sns

# Temporal analysis function
def temporal_analysis(adata2, name, leiden_cluster):
    tmp = adata2.copy()
    tmp.X = tmp.layers['log1p_norm']
    sc.tl.score_genes(tmp, gene_list=['MKI67','TOP2A','UBE2C','BIRC5','CENPF'], score_name='cycling_score')
    adata2.obs['cycling_score'] = tmp.obs['cycling_score'].values
    sc.pl.umap(adata2, color='cycling_score',show =False, save = f"_{name}_subset_cyclingscores.png")
    print("done computing cycling scores")
    sc.pp.neighbors(adata2, use_rep='X_scVI', n_neighbors=15)
    sc.tl.leiden(adata2, key_added='leiden_mal')
    print("done computing malignant leiden")
    sc.pl.umap(adata2, color=['leiden_mal', leiden_cluster],show =False, save=f"_{name}_leiden.png",ncols=1)
    sc.pl.umap(adata2, color=['leiden_mal', leiden_cluster],legend_loc="on data",show =False, save=f"_{name}_leiden on data.png",ncols=1)
    sc.tl.paga(adata2, groups='leiden_mal')
    print("done computing malignant paga")
    sc.pl.paga(adata2,  color=['leiden_mal'],
            save=f"_{name}_paga.png",show =False)
    sc.tl.umap(adata2, init_pos='paga')
    root_cluster = adata2.obs.groupby('leiden_mal')['cycling_score'].median().idxmin()
    root_cell = adata2[adata2.obs['leiden_mal']==root_cluster].obs['cycling_score'].idxmin()
    adata2.uns['iroot'] = int(list(adata2.obs_names).index(root_cell))
    sc.tl.dpt(adata2,n_dcs=10)
    print("done computing malignant dpt")
    
    # cycling score vs pseudotime
    plt.figure(figsize=(4,3))
    plt.scatter(adata2.obs['dpt_pseudotime'], adata2.obs['cycling_score'], s=6, alpha=0.4)
    plt.xlabel('dpt_pseudotime'); plt.ylabel('cycling_score')
    plt.title('cycling_score vs pseudotime')
    plt.tight_layout()
    plt.savefig(f"{name}_cycling_score vs pseudotime.png")
    plt.close()

    from scipy.stats import spearmanr
    # SOX10 vs pseudotime
    sox10_idx=np.where(adata2.var_names == 'SOX10')[0][0]
    sox10_expr=adata2.layers['log1p_norm'][:,sox10_idx]
    if sp.issparse(sox10_expr):
        sox10_expr = np.asarray(sox10_expr.toarray()).ravel()
    else:
        sox10_expr=np.asarray(sox10_expr).ravel()
    pt = np.asarray(adata2.obs['dpt_pseudotime']).astype(float)
    mask = np.isfinite(pt) & np.isfinite(sox10_expr)
    pt_f = pt[mask]
    expr_f = sox10_expr[mask]   
    rho, p = spearmanr(pt_f, expr_f, nan_policy='omit')
    plt.figure(figsize=(4,3))
    plt.scatter(pt_f, expr_f , s=6, alpha=0.4)
    plt.xlabel('dpt_pseudotime'); plt.ylabel('SOX10 (log1p)')
    plt.title(f'SOX10 vs pseudotime (rho={rho:.2f}, p={p:.1e})' )
    plt.tight_layout()
    plt.savefig(f"{name}_SOX10 vs pseudotime.png", dpi=300)
    plt.close()

    # Spearman check
    from scipy.stats import spearmanr
    rho, p = spearmanr(adata2.obs['dpt_pseudotime'].astype(float), adata2.obs['cycling_score'].astype(float), nan_policy='omit')
    print("Spearman(cycling vs dpt):", rho, p)

    sc.pl.umap(adata2, color=['dpt_pseudotime'],ncols=1,show =False, save = f"{name}_dpt_pseudotime.png")
    sc.pl.violin(
        adata2,
        ['dpt_pseudotime'],
        groupby='cluster_annotations',
        rotation=90,
        show=False,
        save=(f"_{name}_dpt_pseudotime_rotated.png")
    )
    adata2.obs.groupby('cluster_annotations')['dpt_pseudotime'].median().sort_values()
    sc.pl.violin(
        adata2,
        ['cycling_score'],
        groupby='cluster_annotations',
        rotation=90,
        show=False,
        save=f"_{name}_proliferating.png"
    )
    sc.pl.violin(
        adata2,
        ['FN1','COL1A1','COL1A2','THBS1','SNAI2'],
        groupby='cluster_annotations',
        rotation=90,
        show=False,
        save=f"_{name}_invasion.png"
    )#Invasion/EMT
    sc.pl.violin(
        adata2,
        ['NGFR','SOX2','JUN','ATF3'],
        groupby='cluster_annotations',
        rotation=90,
        show=False,
        save=f"_{name}_dedifferentiation.png"
    )
    sc.pl.violin(
        adata2,
        ['ZNF423',"SOX10","SOX2","EBF1"],
        groupby='cluster_annotations',
        rotation=90,
        show=False,
        save=f"_{name}_dedifferentiation.png"
    )
    #Repair/dedifferentiation
    #Looking at specific genes
    genes  = [
        'NGFR',   # repair Schwann
        'JUN',    # injury/repair TF
        'SOX2',   # dedifferentiation
        'HOXA10', # PRC2-loss signature
        'FN1',    # ECM/EMT transition
        'MKI67', 'TOP2A', # proliferation
        'CDK6',
        "SOX10","SOX11", #Schwann
        'ZNF423'      
    ]
    gene_mask = adata2.var_names.isin(genes)
    # get expression matrix (use .raw if you stored normalized/logged values there)
    X = adata2.layers['log1p_norm'][:, gene_mask]
    genes_present = adata2.var_names[gene_mask]
    print("present:", genes_present)
    if sp.issparse(X):
        X = X.toarray()

    expr_df = pd.DataFrame(X, index=adata2.obs_names, columns=genes_present)
    expr_df['pseudotime'] = adata2.obs['dpt_pseudotime'].astype(float)
    # sort by pseudotime and compute rolling mean (window size -> adjust)
    expr_df = expr_df.sort_values('pseudotime')
    window = max(5, int(0.05 * expr_df.shape[0]))  # 5% of cells as window (tweakable)
    smoothed = expr_df[genes_present].rolling(window=window, center=True, min_periods=1).mean()
    n=len(genes_present)
    # plot
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2.5 * n), sharex=True)
    for ax, g in zip(axes, genes_present):
        ax.scatter(expr_df['pseudotime'], expr_df[g], s=6, alpha=0.25)
        ax.plot(expr_df['pseudotime'], smoothed[g], linewidth=2)
        ax.set_ylabel(g, rotation=0, labelpad=40, va='center')  

    # title on the top axis
    axes[0].set_title('Marker expression along dpt_pseudotime')
    # rotate only bottom axis labels
    axes[-1].tick_params(axis='x', labelrotation=90)
    axes[-1].set_xlabel('dpt_pseudotime')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)   # leave space for rotated labels
    plt.savefig(f"{name}_Marker_expression_along_pseudotime.png", dpi=300, bbox_inches='tight')

    # bin pseudotime (e.g., 50 bins) and compute mean expression per bin
    n_bins = 50
    expr_df['bin'] = pd.cut(expr_df['pseudotime'], bins=n_bins, labels=False)
    binned = expr_df.groupby('bin')[genes_present].mean()
    # optional: z-score each gene across bins for visualization
    binned_z = (binned - binned.mean()) / binned.std()

    plt.figure(figsize=(6, max(2, len(genes_present)*0.4)))
    sns.heatmap(binned_z.T, cmap='vlag', cbar_kws={'label': 'z-score'})
    plt.xlabel('pseudotime bin (early -> late)')
    plt.ylabel('genes')
    plt.title('Binned mean expression (z-scored) across pseudotime')
    plt.tight_layout()
    plt.savefig(f"{name}_Marker expression along pseudotime Binned z score .png")

    # bin and compute mean + sem
    grouped = expr_df.groupby('bin')
    bin_centers = grouped['pseudotime'].mean()
    means = grouped[genes_present].mean()
    sems = grouped[genes_present].sem()

    plt.figure(figsize=(8, 4))
    for g in genes_present:
        plt.plot(bin_centers, means[g], label=g)
        plt.fill_between(bin_centers, means[g]-sems[g], means[g]+sems[g], alpha=0.2)


    plt.xlabel('dpt_pseudotime')
    plt.ylabel('expression (log1p)')
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.title('Binned mean ± SEM along pseudotime')
    plt.tight_layout()
    plt.savefig(f"{name}_Marker expression along pseudotime Binned mean and sd.png")

    #Look at the immune cells
    genes = ['SEL1L','FCHSD2','DOCK2','ERMN','MEF2C','IGKC','BLNK','PTPRC','CD3D','CD19','MS4A1']
    present = [g for g in genes if g in adata2.var_names]
    missing = [g for g in genes if g not in adata2.var_names]
    print("present:", present)
    print("missing:", missing)

    sc.pl.umap(adata2, 
               color=present+["leiden_mal"], 
               use_raw=False,
               layer="scvi_normalized", 
               ncols=4, show=False, 
               save=f"_{name}_immune markers.png"
               )
    
    sc.pl.umap(
        adata2,
        color=['SOX10',"ZNF423","EBF1","leiden_mal"],#Genes of interest
        ncols=3,layer="scvi_normalized",
        save=f"_{name}_Schwann_cell_markers.png"
    )

    sc.pl.umap(
        adata2,
        color=['SOX10','MPZ','PLP1','MKI67','FN1',"leiden_mal"], #Schwann
        ncols=3,layer="scvi_normalized",
        save=f"_{name}_Schwann_cell_markers.png"
    )
    return(adata2)

#all data
adata=sc.read_h5ad("adata manual annotations2.h5ad")
adata.obs["cluster_annotations"]=adata.obs["leiden_res_scvi_1.00"].astype(str) + adata.obs["Alternate_Overview"].astype(str)
print(adata.obs["cluster_annotations"].unique())
adata=temporal_analysis(adata, name="adata", leiden_cluster="leiden_res_scvi_1.00")
adata.write_h5ad("adata_with_temporal.h5ad", compression="gzip")

#Only malignant clusters
print("Now onto malignant")
malignant = adata[
    adata.obs['Alternate_Overview'] == 'Malignant'
].copy()
malignant=temporal_analysis(malignant,name="Malignant")
malignant.write_h5ad("malignant_pseudo.h5ad", compression="gzip")
#Only non-maligant clusters
print("Now onto non-malignant")
non_mal = adata[
    adata.obs['Alternate_Overview'] != 'Malignant'
].copy()
non_mal=temporal_analysis(non_mal, name="non_mal")
non_mal.write_h5ad("non_mal_pseudo.h5ad", compression="gzip") ##May not have finished writing this
#Without certain clusters
## Getting sense of how many cells per cluster per sample

x=pd.crosstab(
    malignant.obs['sample'],
    malignant.obs['leiden_mal'],
    normalize='index'
)

x.to_csv("Normalizaed number of cells from each sample.csv")

x=pd.crosstab(
    adata.obs['tumor_type'],
    adata.obs['leiden_mal'],
    normalize='index'
)

x.to_csv("Normalizaed number of cells from each tumor type.csv")

x = (
    malignant.obs
    .groupby(['sample', 'cluster_annotations'])['dpt_pseudotime']
    .mean()
    .unstack()
)
x.to_csv("Pseudotime from each sample.csv")

print(malignant.obs.groupby('cluster_annotations')['dpt_pseudotime'].median())



