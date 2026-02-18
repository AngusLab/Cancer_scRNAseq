
import os
import scipy. sparse as sps
import numpy as np
import scanpy as sc
import pandas as pd

#adata=sc.read_h5ad("adata_with_cnv.h5ad")
adata=sc.read_h5ad("adata manual annotations2.h5ad")
print(adata)
print(adata.obs["tumor_type"])
print(adata.obs["Alternate_Overview"])


def make_pseudobulks(adata, sample_col, cluster_col,condition,cell_type, min_cells=10, 
                       random_state=0,
                       outdir="count_edgeR"):
    random=np.random.default_rng(random_state) #random num generator. Like set seed
    os.makedirs(outdir, exist_ok=True) #make directory, and if it already exits, that's ok and doesn't have an error

    if getattr(adata, "raw", None) is not None: #The getattr() function returns the value of the specified attribute from the specified object. object, name of attribute you want to get value from. value to return if it doesn't exist
        df = adata.raw.to_adata()
    elif "counts" in adata.layers: #if adata.raw doesn't exit, then see if adata.layer["counts"] exits
        df = adata.copy()
        df.X = adata.layers["counts"].copy() #copy adata and then copy the counts to df.X
    else:
        df = adata.copy ()

    #Checking that these columns exist. Can probably delete later
    required_cols= [sample_col, cluster_col]
    optional_cols= [condition] #can be another annotaiton column for the future
    for c in required_cols:
        if c not in df.obs.columns:
            raise ValueError (f"adata.obs must contain {c}")
    for c in optional_cols:
        if c not in df.obs.columns:
            raise ValueError (f"adata.obs does not contain {c}")
        
    genes = df.var_names #getting genes
    meta_rows = []
    cluster_tables = {} #empty indices

    cluster = df.obs[cluster_col].unique().tolist() #getting list of all the cluster in the adata ob
    for cl in cluster:
        mask_cl=df.obs[cluster_col]==cl #subseting the cluster and writing true where the value is equal to the cluster
        df_cl=df[mask_cl].copy() #Copying that df and column
        if df_cl.n_obs==0: #if the # of observations in the subset is 0
            continue
        samples = df_cl.obs[sample_col].unique().tolist() #Now subsetting all the samples withint that cluster
        cols=[]
        col_names=[]
        for s in samples:
            mask_s=df_cl.obs[sample_col] ==s #now further subsetting by sample and noting true where the sample col equals s
            cond= df_cl.obs.loc[mask_s, condition].unique() #extracting conidtion
            cond = str(cond[0])
            cell= df_cl.obs.loc[mask_s, cell_type].unique() #Extracting cell type
            cell = str(cell[0])
            cell_idx= np.where(mask_s)[0] #retuns row indx where mask_s is true
            n = len(cell_idx)
            if n < min_cells:
                print (f"Skipping pseudo for sample {s},cluster {cl}: only {n} cells")
                continue
            X=df_cl.X #now making sparse df dense
            sparse=sps.issparse(X)
            if sparse:
                X=X.tocsr()
                summed = np.array(X[cell_idx,:].sum(axis=0)).ravel() 
            else:
                summed = X[cell_idx,:].sum(axis=0) #sums the counts across all selected cells for each gene, giving a gene level count vector for the pseudobulk
            pseudo_name=f"{s}_cl{cl}"
            cols.append(summed)
            col_names.append(pseudo_name)
            #cond = df_cl.obs.loc[mask_s, cluster_col].iloc[0]
            meta_rows.append({
                'pseudo': pseudo_name,
                'org_sample': s,
                'cluster':cl,
                'condition': cond,
                'cell_type':cell,
                "cells_in_pseudo":n
                })
        if len(cols)==0:
            continue
        mat = np.vstack(cols).T
        df_new= pd.DataFrame(mat,index=genes,columns=col_names)
        cluster_tables[cl]=df_new
        df_new.to_csv(os.path.join(outdir, f"cluster_{cl}_counts_leiden_res_1.00.csv"))
    meta_df=pd.DataFrame(meta_rows).set_index('pseudo')
    meta_df.to_csv(os.path.join(outdir, "metadata_leiden_res_1.00.csv"))
    return cluster_tables, meta_df


dict_pbs_med2, meta_med2=make_pseudobulks(adata, "sample", "leiden_res_scvi_1.00",condition="tumor_type",cell_type="Alternate_Overview",
                    min_cells=10, random_state=42,
                       outdir="count_edgeR")


print(type(dict_pbs_med2))
print(meta_med2)
all_counts = pd.concat(dict_pbs_med2.values(), axis=1)
assert all_counts.columns.is_unique, "Duplicate pseudobulk names found!"
all_counts.to_csv("no_pseudo_replciates_counts.csv")
meta_med2.to_csv("metadata_nopseudo.csv")