import scanpy as sc
import pandas as pd
import seaborn as sns

adata=sc.read_h5ad("adata_with_cnv.h5ad")

print(adata)
print(adata.obs['leiden_res_scvi_1.00'])

#Reading in metatdata
df = pd.read_csv('manual_annotations_redo_CAB.csv')
print(df.head)

df['Cluster'] = df['Cluster'].astype(str).str.strip()
metadata_dict = df.set_index('Cluster').to_dict('index')

print(metadata_dict)
adata.obs['high_manual_annotation'] = ""
adata.obs['fine_manual_annotation'] = ''
adata.obs['med_manual_annotation'] = ''
adata.obs['Overview'] = ''
adata.obs['Alternate_Overview'] = ''
adata.obs['leiden_res_scvi_1.00'] = adata.obs['leiden_res_scvi_1.00'].astype(str).str.strip()


for index, row in adata.obs.iterrows():
    sample_id = row['leiden_res_scvi_1.00']
    #print(sample_id)
    if sample_id in metadata_dict:
        sample_metadata = metadata_dict[sample_id]
        adata.obs.loc[index, 'high_manual_annotation'] = sample_metadata.get('Annotation_coarse')
        adata.obs.loc[index, 'med_manual_annotation'] = sample_metadata.get('Annotation')
        adata.obs.loc[index, 'fine_manual_annotation'] = sample_metadata.get('Final_Annotation')
        adata.obs.loc[index, 'Overview'] = sample_metadata.get('Overview')
        adata.obs.loc[index, 'Alternate_Overview'] = sample_metadata.get('Alternate_overview')
    else:
        adata.obs.loc[index, 'high_manual_annotation']='Unannotated'
        adata.obs.loc[index, 'med_manual_annotation']='Unannotated'
        adata.obs.loc[index, 'fine_manual_annotation']='Unannotated'

# Verify the update
print(adata.obs[['sample','tumor_type','Overview',"fine_manual_annotation",'leiden_res_scvi_1.00']].head())
missing_clusters = sorted(set(adata.obs['leiden_res_scvi_1.00'].unique()) - set(metadata_dict.keys()))
print("Clusters present in adata but missing from CSV (post-normalization):", missing_clusters[:30])

adata.write_h5ad("adata manual annotations2.h5ad", compression="gzip")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex


def make_pallete(adata, var, max_base=60):

    n=int(adata.obs[var].nunique())
    pool=list(plt.cm.tab20.colors)+list(plt.cm.tab20b.colors)+list(plt.cm.tab20c.colors)
    if n <=len(pool):
        palette=pool[:n]
    else:
        cmap=plt.cm.get_cmap('hsv',n)
        palette=[cmap(i) for i in range(n)]
    
    palette_hex=[to_hex(c) for c in palette]
    adata.uns[f"{var}_colors"]=palette_hex

    return palette_hex

vars_to_pallete=["Alternate_Overview","high_manual_annotation",
                 "med_manual_annotation","fine_manual_annotation",
                 "Overview"]

for v in vars_to_pallete:
    make_pallete(adata, v)


sc.pl.umap(
    adata,
    color=["Overview","tumor_type","ZNF423"],
    ncols=1,
    show = False,
    frameon=False,
    cmap=sns.blend_palette(["lightgray", sns.xkcd_rgb["red"]], as_cmap=True),
    save="_overview_annotation_and tumor type_with_ZNF423_updated.png",
    layer= "scvi_normalized"
)

sc.pl.umap(
    adata,
    color=["high_manual_annotation","ZNF423"],
    ncols=1,
    show = False,
    frameon=False,
    cmap=sns.blend_palette(["lightgray", sns.xkcd_rgb["red"]], as_cmap=True),
    save="_annotation_with_ZNF423_updated.png",
    layer= "scvi_normalized"
)


sc.pl.umap(
    adata,
    color=["high_manual_annotation"],
    ncols=1,
    show = False,
    frameon=False,
    save="_annotation.png",
    title="Cell Annotations"
)

sc.pl.umap(
    adata,
    color=["Alternate_Overview"],
    ncols=1,
    show = False,
    frameon=False,
    save="_Alternate_Overview.png",
    title="Cell Annotations",
    size=2
)

sc.pl.umap(
    adata,
    color=["Overview"],
    ncols=1,
    show = False,
    frameon=False,
    save="_Overview_size2.png",
    title="Cell Annotations",
    size=2
)


sc.pl.umap(
    adata,
    color=["high_manual_annotation"],
    ncols=1,
    show = False,
    frameon=False,
    save="_high_manual_annotation_size2.png",
    layer= "scvi_normalized",
    title="Cell Annotations",
    size=2
)

sc.pl.umap(
    adata,
    color=["fine_manual_annotation"],
    ncols=1,
    show = False,
    frameon=False,
    save="_indepth annotation.png",
    layer= "scvi_normalized",
    title = "In depth annotation",
    size=2
)


sc.pl.umap(
    adata,
    color=["leiden_res_scvi_1.00"],
    ncols=1,
    legend_loc="on data",
    show = False,
    frameon=False,
    save="clustering on data.png",
    layer= "scvi_normalized",
    title="Clustering Resolution 1.00"

)

genes= ["ZNF423","SOX10","EBF1", "KI67","PCNA","HMGA2","AXL", "EGFR","EPHA3","CDK19","DLK1","SOX9","SOX11","S100A1","S100B"]
genes_present = [g for g in genes if g in adata.var_names]
print("Genes found:", genes_present)
for g in genes_present:
    sc.pl.umap(
        adata,
        color=g,
        show = False,
        frameon=False,
        cmap=sns.blend_palette(["lightgray", sns.xkcd_rgb["red"]], as_cmap=True),
        save=f"_{g}_size2.png",
        layer= "scvi_normalized",
        size = 2
    )
    sc.pl.umap(
        adata,
        color=g,
        ncols=1,
        show = False,
        frameon=False,
        cmap=sns.blend_palette(["lightgray", sns.xkcd_rgb["red"]], as_cmap=True),
        save=f"_{g}_size2_log1p.png",
        layer= "log1p_norm",
        size = 2
    )


