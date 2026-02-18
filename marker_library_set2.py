import scanpy as sc
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import scipy.sparse as sp
from typing import Dict


def setup_logger(out_prefix):
    Path(out_prefix).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(out_prefix, "markers.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler() 
        ]
    )

def caf_markers(adata,     
                outdir= "CAFPercentage",
                outdir2="UMAPs_for_each_CAF_class",
                cluster_key = "leiden_res_scvi_1.00",
                layer = "log1p_norm"):

    #Setting up output and directory
    Path(outdir).mkdir(parents=True, exist_ok=True)
    #make and set directory
    setup_logger(outdir)
    sc.settings.figdir = os.path.abspath(outdir)

    # CAF panels (cleaned)
    myCAF = ["ACTA2","TAGLN","MYL9","CNN1","MYH11","CALD1","PDGFRB","ITGB1","TGFBI"]
    iCAF = ["IL6","CXCL12","CCL2","CXCL1","LIF","PTGS2","IL1R1","VCAM1","STAT3"]
    apCAF = ["HLA-DRA","HLA-DRB1","HLA-DPA1","CD74","CIITA","HLA-DQA1","HLA-DPB1"]
    bmCAF = ["LAMA2","LAMB1","COL1A1","COL1A2","COL3A1","COL4A1","COL4A2","COL5A2","COL6A1","COL6A3","POSTN","SPARC","FN1"]
    invCAF = ["COL11A1","ADAM12","ADAMTS9","ADAMTSL3","LOXL2","TNC","THSD4","POSTN"]
    pdgfra_CAF = ["PDGFRA","COL1A1","LAMA2","PDGFC","IGF1","FAP"]
    abca_CAF = ["ABCA6","ABCA8","ABCA9","ABCA10","IGFBP7","SLC27A2"]
    sCAF = ["HSPA5","HSP90AA1","CALR","MT2A","SPP1","TIMP1"]   # removed RPLP1,RPS27,MT-CO1
    periCAF = ["MCAM","RGS5","CSPG4","PDGFRB","ACTA2","ANPEP"]
    prolifCAF = ["MKI67","TOP2A"]  # removed RPS27,RPLP1
    angio = ["VEGFA","PDGFC","ANGPTL4","CXCL12","HGF"]

    # separate stress/ribo signature (if you want to track it)
    stress_signature = ["RPLP1","RPS27","MT-CO1","MT-ND1","HSPA5","HSP90AA1"]

    marker={
        "myCAF":myCAF,
        "inflammatory CAF":iCAF,
        "antigen-presenting CAF":apCAF,
        "ECM-CAF":bmCAF,
        "Invasive CAF":invCAF,
        "PDGFRA intersitial ECM-CAF":pdgfra_CAF,
        "ABCA CAF":abca_CAF,
        "Proliferative CAF":prolifCAF,
        "Angiogenesis CAF": angio,
        "stress CAF":sCAF,
        "Perivascular CAF":periCAF ,
        "Stressed signature": stress_signature
    }

    os.makedirs(outdir, exist_ok=True) 
    os.makedirs(outdir2, exist_ok=True)

    def present(genes, adata):
        present = []
        missing = []
        for g in genes:
            if g in adata.var_names:
                present.append(g)
            else:
                missing.append(g)
        return present, missing


    def filter_panels_to_adata(panels:dict, adata, drop_empty=True, verbose = True):
        filtered={}
        for name, genes in panels.items():
            if not isinstance(genes, (list,tuple)):
                if verbose:
                    print(f"Skipping panel {name}: not a list/tuple")
                continue
            pres, miss=present(genes, adata)
            if verbose:
                logging.info(f"{name}:{len(pres)}/{len(genes)} present.")
            if len(pres)>0 or not drop_empty:
                filtered[name]=pres
        return filtered

    def add_module_score(adata, gene_list, score_name):
        genes_present, _ = present(gene_list, adata)
        if len(genes_present) == 0:
            logging.info(f"No genes present for {score_name}; skipping.")
            return
        # use scanpy's score_genes (it creates '{score_name}_score' in .obs)
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=score_name, use_raw=False, layer=layer)
        logging.info(f"Added score: {score_name} (n_genes={len(genes_present)})")


    def score_marker_panels2(adata, marker_panels: Dict[str, list], groupby=cluster_key, use_raw=False, module_score_prefix="score_"):
        
        #Get expression matrix and variables names
        X = adata.layers[layer]
        var_names = np.array(adata.var_names)
        #Cluster
        clusters = adata.obs[groupby].unique()

        results = []
        #Get genes and their locations in var_names
        for panel_name, genes in marker_panels.items():
                valid_genes = [g for g in genes if g in var_names]
                if len(valid_genes) == 0:
                    logging.warning(f"[{panel_name}] no valid genes found - skipping")
                    continue

                # Subset X to panel genes
                gene_idx = [adata.var_names.get_loc(g) for g in valid_genes]
                #Get gene by cluster percent-expressing table
                pct_df=pd.DataFrame(index=clusters, columns=valid_genes,dtype=float)

                # percent and mean per cluster 
                for cl in clusters:
                    cell_mask = (adata.obs[groupby] == cl).values
                    idx_cells=np.where(cell_mask)[0]

                    if idx_cells.size==0:
                        pct_df.loc[cl,:]=np.nan
                        continue
                    #Subset X copy to panel genes
                    sub=X[idx_cells][:,gene_idx]
                    #make less sparse
                    if sp.issparse(sub):
                        arr=sub.toarray()
                    else:
                        arr=np.asarray(sub)
                    
                    # Percent expressing (>0)
                    per_gene_pct = (arr > 0).sum(axis=0) / arr.shape[0] * 100
                    pct_df.loc[cl, valid_genes]=per_gene_pct
                    mean_percent=float(np.nanmean(per_gene_pct)) 

                    
                    #Percent of cells expresing any gene in the panel
                    mask=(adata.obs[groupby]==cl).values
                    idx_cells=np.where(mask)[0]
                    if idx_cells.size == 0:
                        pct_any=np.nan
                        mean_expr=np.nan
                    else:
                        sub_any=X[idx_cells][:,gene_idx]
                        arr_any=sub_any.toarray() if sp.issparse(sub_any) else np.asarray(sub_any)
                        pct_any=float((arr_any>0).any(axis=1).mean()*100.0)
                        mean_expr=float(np.nanmean(arr_any)) if arr_any.size else np.nan

                               
                    module_score = adata.obs.loc[adata.obs[groupby]==cl, f"score_{panel_name}"].mean()
                    
                    results.append({
                    "cluster": cl,
                    "panel": panel_name,
                    "n_genes": len(valid_genes),
                    "mean_percent_across_genes": mean_percent,
                    "percent_cells_any_gene": pct_any,
                    "mean_expr": mean_expr,
                    "module_score": module_score
                })

                pct_df.to_csv(os.path.join(outdir,f"Percentage of marker genes for {panel_name} per cluster.csv"))
                logging.info(f"[{panel_name}] wrote percent-expressing CSV (shape: {pct_df.shape}).")
        panel_df = pd.DataFrame(results)
        panel_df.to_csv(os.path.join(outdir,f"Summary of marker genes per cluster for CAFs.csv"))
        return panel_df

    #Run functions
    marker_list_filtered= filter_panels_to_adata(marker, adata)
    for name, genes in marker_list_filtered.items():
        add_module_score(adata, genes, "score_"+name)

    # combine everything (need to run the above to run this one)
    score_marker_panels2(adata, marker_list_filtered, groupby=cluster_key)
    logging.info("Finished running functions")
    #Make score a df and z score it
    scores_cols=[f"score_{k}" for k in marker_list_filtered.keys() if len(marker_list_filtered[k])>0]
    scores_df=adata.obs[scores_cols].copy()
    row_std=scores_df.std(axis=1).replace(0,0.1)
    scores_z=(scores_df.sub(scores_df.mean(axis=1), axis=0)).div(row_std, axis=0)
    scores_z.columns=[c+"_z" for c in scores_cols]
    adata.obs=adata.obs.join(scores_z)

    #Assign best subtype per cell for based on z-score
    zcols=[c for c in adata.obs.columns if c.endswith("_z")]
    adata.obs["CAF_best_subtype"]=adata.obs[zcols].idxmax(axis=1).str.replace("_z","").str.replace("score_","")
    adata.obs["CAF_best_score"]=adata.obs[zcols].max(axis=1)
    adata.obs["CAF_second_score"]=adata.obs[zcols].apply(lambda r : r.nlargest(2).values[-1], axis=1)
    adata.obs["CAF_confidence"]=adata.obs["CAF_best_score"]-adata.obs["CAF_second_score"]

    logging.info("beginning plotting")
    #plotting
    sc.pl.umap(adata,color = "CAF_best_subtype",layer = layer, title = "Best CAF per cluster", save = "best CAF.png", show = False)
    plot_scores = scores_cols
    sc.pl.umap(adata, color=plot_scores, cmap='viridis', layer = layer, ncols=3,save = "best CAF_scores.png", size=20, show = False)
    #Per cluster majority
    cluster_subtype=adata.obs.groupby(cluster_key)["CAF_best_subtype"].agg(lambda x:x.value_counts().idxmax())
    cluster_subtype_conf=adata.obs.groupby(cluster_key)["CAF_confidence"].mean()
    cluster_summary=pd.DataFrame({
        "n_cells":adata.obs.groupby(cluster_key).size(),
        "majority_CAF_subtype": cluster_subtype,
        "mean_subtype_confidence":cluster_subtype_conf
    })
    
    cluster_summary.to_csv(os.path.join(outdir,f"cluster_CAF_subtype_summary.csv"))
    print(cluster_summary.sort_values("n_cells", ascending=False).head(30))

    df=adata.obs.groupby([cluster_key,"sample"]).size().unstack(fill_value=0)
    df.to_csv(os.path.join(outdir,f"Samples per cluster.csv"))
    df.head()

    #Dotplot of substypes
    viz_genes=sorted(list({g for genes in marker_list_filtered.values() for g in genes}))
    viz_genes=[g for g in viz_genes if g in adata.var_names][:60]
    sc.pl.dotplot(adata, var_names=viz_genes, groupby=cluster_key, cmap="Reds", layer = layer, save = "_CAF_subtypes.png")

    logging.info("Plotting for each individual CAF")
    sc.settings.autoshow = False
    sc.settings.figdir = os.path.abspath(outdir2)
    #UMAP for each marker selection and individual UMAPs for each gene
    for ct, genes in marker_list_filtered.items():
        genes_present = [g for g in genes if g in adata.var_names]
        base_name = ct.replace(" ", "_").replace("/", "_")
        outdir3 = os.path.join(outdir2, base_name)
        os.makedirs(outdir3,exist_ok=True)
        sc.settings.figdir = os.path.abspath(outdir3)
        sc.pl.dotplot(adata, var_names=genes_present, groupby=cluster_key, layer = layer,
                    standard_scale="var", swap_axes=True, title=f"{ct} _{cluster_key}", save= f"_{base_name}.png")
        sc.pl.violin(adata, keys=genes_present, groupby=cluster_key,layer = layer, stripplot=False,save=f"_{base_name}_genes.png")
        sc.pl.umap(adata, color=genes_present, vmax=3,layer = layer, save =  f"_{base_name}_genes2.png")

    logging.info("All done")



def markers(adata,
            outdir= "Marker_Percentage",
            outdir2="UMAPs_for_each_marker_class",
            cluster_key = "leiden_res_scvi_1.00",
            layer="log1p_norm",
            layer_umap = "scvi_normalized"):
    
        #Setting up output and directory
    Path(outdir).mkdir(parents=True, exist_ok=True)
    #make and set directory
    setup_logger(outdir)
    sc.settings.figdir = os.path.abspath(outdir)

    panels = {
        # Immune lineage anchors
    # ---------- Stromal / neural ----------
    "Schwann": ["SOX10","S100B","ERBB3","NGFR","GFRA1","PLP1","MPZ","PMP22","MBP","EGR2","L1CAM"],
    "Fibroblast_generic": ["PDGFRA","FAP","THY1","DCN","LUM","COL1A1","COL1A2","COL3A1","SPARC","FN1","POSTN","MMP2","PDPN"],
    "Pericyte_SMC": ["RGS5","CSPG4","MCAM","NOTCH3","ACTA2","TAGLN","MYL9","CNN1","MYH11","PDGFRB","DES"],
    "Endothelial": ["PECAM1","VWF","CDH5","KDR","FLT1","TEK","PLVAP","RAMP2"],
    "Adipocyte": ["ADIPOQ","FABP4","PLIN1","PPARG","LPL","CIDEC","LIPE"],
    "Myocyte_striated": ["CKM","TNNT1","TPM2","MYH1","MYH2","MYH7"],
    # ---------- Tumor / cycling ----------
    "Tumor_onco": ["EGFR","PDGFRA","MET","MYC","MDM2","TERT"],
    "Proliferation": ["MKI67","TOP2A","PCNA","TYMS","MCM5"],
    # ---------- Immune backbone ----------
    "Immune_pan": ["PTPRC"],
    "T_cell_core": ["TRAC","TRBC1","CD3D","CD3E","LCK","TRAT1","CD247"],
    "Treg_CD4": ["CD4","IL2RA","FOXP3","CTLA4","IKZF2","TIGIT"],
    "CD8_cytotoxic_exhausted": ["CD8A","NKG7","PRF1","GZMB","PDCD1","TOX","LAG3"],
    "NK_ILC": ["NKG7","GNLY","KLRD1","FCGR3A","PRF1","GZMB","TRDC"],
    "B_cell": ["MS4A1","CD79A","CD79B","CD74","HLA-DRA","BANK1"],
    "Plasma_cell": ["MZB1","XBP1","JCHAIN","IGHG1","SDC1"],
    # ---------- Myeloid / macrophage ----------
    "Myeloid_core": ["LYZ","LST1","TYROBP","FCER1G","CTSS","S100A8","S100A9"],
    "Macrophage_TAM_M2like": ["CD68","CSF1R","MERTK","C1QA","C1QB","APOE","CD163","MRC1","TREM2"],
    "Macrophage_inflammatory_M1like": ["IL1B","TNF","CXCL9","CXCL10","CCL2","NFKBIA"],
    # ---------- Dendritic / antigen presentation ----------
    "DC_MHCII_APC": ["HLA-DRA","HLA-DRB1","CD74","CIITA","ITGAX","FLT3"],
    "cDC1": ["CLEC9A","XCR1","BATF3","IRF8","CADM1"],
    "cDC2": ["CD1C","FCER1A","CLEC10A","IRF4","SIRPA"],
    "pDC": ["CLEC4C","IL3RA","LILRA4","TCF4","GZMB"],
    "DC_maturation_migration": ["CCR7","LAMP3","FSCN1"],
    "IFN_ISG": ["ISG15","IFIT1","IFIT3","MX1","OAS1","STAT1","IRF7"],
    "moDC_like": ["LYZ","LST1","FCGR3A","S100A8","S100A9"],
    # ---------- Rare lineages ----------
    "Mast_cell": ["TPSAB1","TPSB2","KIT","MS4A2","CPA3"],
    "Erythroid": ["HBB","HBA1","HBA2","ALAS2"],
    "Platelet": ["PPBP","PF4","ITGA2B"],
    }


    marker_list = panels

    os.makedirs(outdir, exist_ok=True) 
    os.makedirs(outdir2, exist_ok=True)

    def present(genes, adata):
        present = []
        missing = []
        for g in genes:
            if g in adata.var_names:
                present.append(g)
            else:
                missing.append(g)
        return present, missing

    def filter_panels_to_adata(panels:dict, adata, drop_empty=True, verbose = True):
        filtered={}
        for name, genes in panels.items():
            if not isinstance(genes, (list,tuple)):
                if verbose:
                    print(f"Skipping panel {name}: not a list/tuple")
                continue
            pres, miss=present(genes, adata)
            if verbose:
                logging.info(f"{name}:{len(pres)}/{len(genes)} present.")
            if len(pres)>0 or not drop_empty:
                filtered[name]=pres
        return filtered
    
        # function to add scores to adata.obs
    def add_module_score(adata, gene_list, score_name):
        genes_present, _ = present(gene_list, adata)
        if len(genes_present) == 0:
            logging.info(f"No genes present for {score_name}; skipping.")
            return
        # use scanpy's score_genes (it creates '{score_name}_score' in .obs)
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=score_name, use_raw=False, layer=layer)
        logging.info(f"Added score: {score_name} (n_genes={len(genes_present)})")


    def score_marker_panels2(adata, marker_panels: Dict[str, list], groupby=cluster_key):
        
        #Get expression matrix and variables names
        X = adata.layers[layer]
        var_names = np.array(adata.var_names)
        #Cluster
        clusters = adata.obs[groupby].unique()

        results = []
        #Get genes and their locations in var_names
        for panel_name, genes in marker_panels.items():
                valid_genes = [g for g in genes if g in var_names]
                if len(valid_genes) == 0:
                    logging.warning(f"[{panel_name}] no valid genes found - skipping")
                    continue

                # Subset X to panel genes
                gene_idx = [adata.var_names.get_loc(g) for g in valid_genes]
                #Get gene by cluster percent-expressing table
                pct_df=pd.DataFrame(index=clusters, columns=valid_genes,dtype=float)

                # percent and mean per cluster 
                for cl in clusters:
                    cell_mask = (adata.obs[groupby] == cl).values
                    idx_cells=np.where(cell_mask)[0]

                    if idx_cells.size==0:
                        pct_df.loc[cl,:]=np.nan
                        continue
                    #Subset X copy to panel genes
                    sub=X[idx_cells][:,gene_idx]
                    #make less sparse
                    if sp.issparse(sub):
                        arr=sub.toarray()
                    else:
                        arr=np.asarray(sub)
                    
                    # Percent expressing (>0)
                    per_gene_pct = (arr > 0).sum(axis=0) / arr.shape[0] * 100
                    pct_df.loc[cl, valid_genes]=per_gene_pct
                    mean_percent=float(np.nanmean(per_gene_pct)) 

                    
                    #Percent of cells expresing any gene in the panel
                    mask=(adata.obs[groupby]==cl).values
                    idx_cells=np.where(mask)[0]
                    if idx_cells.size == 0:
                        pct_any=np.nan
                        mean_expr=np.nan
                    else:
                        sub_any=X[idx_cells][:,gene_idx]
                        arr_any=sub_any.toarray() if sp.issparse(sub_any) else np.asarray(sub_any)
                        pct_any=float((arr_any>0).any(axis=1).mean()*100.0)
                        mean_expr=float(np.nanmean(arr_any)) if arr_any.size else np.nan

                               
                    module_score = adata.obs.loc[adata.obs[groupby]==cl, f"score_{panel_name}"].mean()
                    
                    results.append({
                    "cluster": cl,
                    "panel": panel_name,
                    "n_genes": len(valid_genes),
                    "mean_percent_across_genes": mean_percent,
                    "percent_cells_any_gene": pct_any,
                    "mean_expr": mean_expr,
                    "module_score": module_score
                })

                pct_df.to_csv(os.path.join(outdir,f"Percentage of marker genes for {panel_name} per cluster.csv"))
                logging.info(f"[{panel_name}] wrote percent-expressing CSV (shape: {pct_df.shape}).")

        panel_df = pd.DataFrame(results)
        panel_df.to_csv(os.path.join(outdir,f"Summary of marker genes per cluster.csv"))
        #Get rank of top 100 genes in case there is a wonky cluster that doesn't fit any classification
        sc.tl.rank_genes_groups(adata, groupby=cluster_key, layer=layer, method = "wilcoxon", key_added="rank_genes_res_1.00")
        de_05_df = sc.get.rank_genes_groups_df(adata, key = "rank_genes_res_1.00", group=None)
        top100=(
            de_05_df
            .groupby('group')
            .head(100)
            .reset_index(drop=True)
        )
        top100.to_csv(os.path.join(outdir,f"top100_genes_per_cluster_res_1.0.csv"),index=False)

        return panel_df

    #Run functions
    marker_list_filtered= filter_panels_to_adata(marker_list, adata)

    for name, genes in marker_list.items():
        add_module_score(adata, genes, "score_"+name)

    # combine everything (need to run the above to run this one)
    score_marker_panels2(adata, marker_list_filtered, groupby=cluster_key)
    logging.info("Finished running functions")
    #Make score a df and z score it
    scores_cols=[f"score_{k}" for k in marker_list_filtered.keys() if len(marker_list_filtered[k])>0]
    scores_df=adata.obs[scores_cols].copy()
    row_std=scores_df.std(axis=1).replace(0,0.1)
    scores_z=(scores_df.sub(scores_df.mean(axis=1), axis=0)).div(row_std, axis=0)
    scores_z.columns=[c+"_marker_z" for c in scores_cols]
    adata.obs=adata.obs.join(scores_z)

    #Assign best subtype per cell for based on z-score
    zcols=[c for c in adata.obs.columns if c.endswith("_marker_z")]
    adata.obs["Marker_best_subtype"]=adata.obs[zcols].idxmax(axis=1).str.replace("_marker_z","").str.replace("score_","")
    adata.obs["Marker_best_score"]=adata.obs[zcols].max(axis=1)
    adata.obs["Marker_second_score"]=adata.obs[zcols].apply(lambda r : r.nlargest(2).values[-1], axis=1)
    adata.obs["Marker_confidence"]=adata.obs["Marker_best_score"]-adata.obs["Marker_second_score"]
    cluster_subtype=adata.obs.groupby(cluster_key)["Marker_best_subtype"].agg(lambda x:x.value_counts().idxmax())
    cluster_subtype_conf=adata.obs.groupby(cluster_key)["Marker_confidence"].mean()
    cluster_summary=pd.DataFrame({
        "n_cells":adata.obs.groupby(cluster_key).size(),
        "majority_Marker_subtype": cluster_subtype,
        "mean_subtype_confidence":cluster_subtype_conf
    })

    cluster_summary.to_csv(os.path.join(outdir,f"cluster_Marker_subtype_summary.csv"))
    logging.info(cluster_summary.sort_values("n_cells", ascending=False).head(30))

    df=adata.obs.groupby([cluster_key,"sample"]).size().unstack(fill_value=0)
    df.to_csv(os.path.join(outdir,f"Samples per cluster.csv"))
    logging.info(df.head())

    sc.settings.autoshow = False
    # UMAP colored by module scores 
    sc.pl.umap(adata, color=[f"score_{k}" for k in marker_list.keys() if f"score_{k}" in adata.obs.columns]+[cluster_key,"tumor_type"],
            cmap='viridis',wspace=0.5, ncols=3,layer=layer_umap, save = (f"_of markers by {cluster_key}.png"), show = False)
    sc.pl.umap(adata, color=[f"score_{k}" for k in marker_list.keys() if f"score_{k}" in adata.obs.columns]+[cluster_key,"tumor_type"],
            cmap='viridis',wspace=0.5, ncols=3,layer=layer, save = (f"_of markers by {cluster_key}_log1p.png"), show = False)


    # dotplot of hallmark genes across clusters 
    viz_genes=sorted(list({g for genes in marker_list_filtered.values() for g in genes}))
    viz_genes=[g for g in viz_genes if g in adata.var_names][:60]
    sc.pl.dotplot(adata, var_names=viz_genes, groupby=cluster_key, standard_scale='var',layer=layer, save=(f"_markers_by_{cluster_key}.png"), show = False)
    logging.info("Plotting for each individual marker")
    
    sc.settings.figdir = os.path.abspath(outdir2)
    #UMAP for each marker selection and individual UMAPs for each gene
    for ct, genes in marker_list_filtered.items():
        genes_present = [g for g in genes if g in adata.var_names]
        base_name = ct.replace(" ", "_").replace("/", "_")
        outdir3 = os.path.join(outdir2, base_name)
        os.makedirs(outdir3,exist_ok=True)
        sc.settings.figdir = os.path.abspath(outdir3)
        sc.pl.dotplot(adata, var_names=genes_present, groupby=cluster_key, layer = layer,
                    standard_scale="var", swap_axes=True, title=f"{ct} _{cluster_key}", save= f"_{base_name}.png")
        sc.pl.violin(adata, keys=genes_present, groupby=cluster_key, stripplot=False,save=f"_{base_name}_genes.png")
        sc.pl.umap(adata, color=genes_present, vmax=3,layer = layer_umap, save =  f"_{base_name}_genes.png")
        sc.pl.umap(adata, color=genes_present, vmax=3,layer = layer, save =  f"_{base_name}_genes_log1p.png")