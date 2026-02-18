import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.stats import t as tdist
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import scanpy as sc
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
try:
    import statsmodels.api as sm
    _has_sm = True
except Exception:
    _has_sm = False

#Functions used in the other functions
def compute_cycling_score(
        adata,
        layer = "log1p_norm",
        score_name="cell_cycle_score"
):
    s_genes_human = [
        "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
        "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1",
        "GMNN","WDR76"
    ]
    g2m_genes_human = [
        "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
        "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
        "CENPE","CTCF"
    ]

    if layer is not None and layer in adata.obs:
        mat=adata.layer[layer]
    else:
        print(f"{layer} is not in adata.layer")
    
    if sp.issparse(mat):
        mat=mat.toarray()
    
    import anndata
    tmp=anndata.AnnData(X=mat, obs=adata.obs.copy(), var=adata.var.copy())
    sc.tl.score_genes_cell_cycle(tmp, s_genes=s_genes_human, g2m_genes=g2m_genes_human, use_raw=False)
    adata.obs["S_score"]=tmp.obs["S_score"].values
    adata.obs["G2M_score"]=tmp.obs["G2M_score"].values

    adata.obs[score_name]=adata.obs["S_score"].fillna(0)+adata.obs["G2M_score"].fillna(0)
    print(f"Wrote cycling score to adata.obs['{score_name}'].")
    return score_name

def regress_out_covariate_lstsq(mat, cov):
    """
    Regress out covariate from mat columns and from cov itself. mat: (n_cells, n_genes) or (n_cells, 1). cov: (n_cells,) or (n_cells, n_cov)
    Returns residuals_mat (n_cells, n_genes) and resid_cov (n_cells,)
    Using linear least squares closed form (X^T X)^{-1} X^T Y with intercept.
    """
    mat=np.asarray(mat, dtype=float)
    cov=np.asarray(cov, dtype=float)

    #normalize cov to 2D
    single_cov=False
    if cov.ndim==1:
        cov=cov.reshape(-1,1)
        single_cov=True
    elif cov.ndim !=2:
        raise ValueError ("cov must be 1D or 2D")
    
    if cov.shape[0] != mat.shape[0]:
        raise ValueError ("mat and cov must have the same number of rows aka cells")


    valid = np.all(np.isfinite(cov), axis=1)
    if valid.sum() < 3:
        raise ValueError("Not enough valid covariate values (rows) to regress out.")
    
    #design matrix (intercept + covariates)
    X = np.column_stack([np.ones(valid.sum()), cov[valid,:]]) # (n_valid,1+n_covs)
    Y=mat[valid,:] #n_valid, n_genes
    #solve with lstsq
    betas,*_=np.linalg.lstsq(X,Y,rcond=None) #(1+n_covs, n_genes)
    fitted=X.dot(betas) #(n_valid,n_genes)
    resid_mat=np.full_like(mat,np.nan,dtype=float)
    resid_mat[valid,:]=Y-fitted

    #residulaize covariates themselves
    betas_cov,*_=np.linalg.lstsq(X,cov[valid,:],rcond=None) #(1+ n_covs, n_covs)
    fitted_cov=X.dot(betas_cov) #(n_valid, n_covs)
    resid_cov=np.full_like(cov, np.nan, dtype=float)
    resid_cov[valid,:]=cov[valid,:]-fitted_cov

    if single_cov:
        resid_cov=resid_cov.ravel()
    
    return resid_mat, resid_cov
 

def benjamini_hochberg(pvals):
    """Return BH-FDR-adjusted p-values for 1D array of pvals (NaNs preserved)."""
    p = np.array(pvals, dtype=float) #Convert to a numpy array
    n = (~np.isnan(p)).sum() #Count how many tests should be run aka where it's not NAs
    q = np.full_like(p, np.nan, dtype=float) #Initialize output array with nas preserved
    if n == 0:
        return q
    idx = np.where(~np.isnan(p))[0] #Only valid p values
    p_nonan = p[idx] #returning index of non na p values
    order = np.argsort(p_nonan) #orders the p values from smallest to largest
    ranked = np.empty_like(order) #create unintialized array with the same shape and type as order
    ranked[order] = np.arange(len(order)) #restore original order after correction
    # BH:
    denom = np.arange(1, len(p_nonan) + 1)[::-1] #measures size of pnonan
    numer = p_nonan[order][::-1] * len(p_nonan) / denom #BH formula
    adj_rev = np.minimum.accumulate(numer)  # monotonic. Makes sure q values never decrease as p values icnrease
    adj = adj_rev[::-1]
    adj = np.minimum(adj, 1.0)
    q[idx] = adj[ranked] #Puts 1 values back into original position
    return q

def compute_spearman_vectorized(mat, vec):
    """
    Compute Spearman correlation between vec (n_cells,) and each column of mat (n_cells, n_genes).
    Returns arrays (rho, pval).
    Uses ranks and Pearson formula for speed.
    """
    n_cells = mat.shape[0]
    if mat.shape[0] != vec.shape[0]:
        raise ValueError("mat and vec must have same number of rows (cells).")
    # rank
    rank_vec = rankdata(vec, method='average') #Assigns ranks to vec, which is pseudotime
    try:
        rank_mat = rankdata(mat, axis=0, method='average') #Assigns rank to mat, which is genes
    except TypeError:
        # Older scipy fallback
        rank_mat = np.apply_along_axis(rankdata, 0, mat, method='average')
    # center
    rv_c = rank_vec - rank_vec.mean()
    rm_c = rank_mat - rank_mat.mean(axis=0) #Centers around 0
    num = np.dot(rv_c, rm_c)  # (n_genes,)
    denom = np.sqrt(np.sum(rv_c**2) * np.sum(rm_c**2, axis=0)) #Normalizes correlation to lie in [-1,1]
    with np.errstate(divide='ignore', invalid='ignore'): #spearman correlationg coefficients
        rho = num / denom
    rho = np.clip(rho, -1.0, 1.0)
    # p-values via t-stat
    df = n_cells - 2
    t_stat = rho * np.sqrt(np.divide(df, 1.0 - rho**2, out=np.full_like(rho, np.nan), where=(1.0-rho**2)!=0))
    pvals = 2.0 * stats.t.sf(np.abs(t_stat), df) #p value
    pvals[np.isnan(rho)] = np.nan
    return rho, pvals

def compute_pearson_vectorized(mat, vec):
    """Pearson correlation vectorized between vec and each column of mat."""
    v = vec - vec.mean() #Center data
    m = mat - mat.mean(axis=0) #Center data
    num = np.dot(v, m) #Pearson formulat
    denom = np.sqrt(np.sum(v**2) * np.sum(m**2, axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / denom
    r = np.clip(r, -1.0, 1.0)
    df = mat.shape[0] - 2
    t_stat = r * np.sqrt(np.divide(df, 1.0 - r**2, out=np.full_like(r, np.nan), where=(1.0-r**2)!=0))
    pvals = 2.0 * stats.t.sf(np.abs(t_stat), df)
    pvals[np.isnan(r)] = np.nan
    return r, pvals

import numpy as np

def residualize_matrix_on_covariate(mat, cov):
    """
    Regress each column of mat (n_cells, n_genes) on cov (n_cells,) or cov (n_cells, n_covs)
    using normal-equation closed-form and return residuals of same shape (n_cells, n_genes).
    Inputs:
      mat: ndarray shape (n_cells, n_genes) or (n_cells,) for single gene
      cov: ndarray shape (n_cells,) or (n_cells, n_covs)
    Returns:
      resid: ndarray shape (n_cells, n_genes)
      betas: ndarray shape (n_covs+1, n_genes)
    Notes:
      - Uses intercept automatically.
      - Defensive about shapes (coerces 1D -> 2D where needed).
    """
    mat = np.asarray(mat)
    cov = np.asarray(cov)

    # coerce mat to 2D (n_cells, n_genes)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    elif mat.ndim != 2:
        raise ValueError("mat must be 1D or 2D")

    n_cells = mat.shape[0]

    # coerce cov to 2D (n_cells, n_covs)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    elif cov.ndim != 2:
        raise ValueError("cov must be 1D or 2D")

    if cov.shape[0] != n_cells:
        raise ValueError(f"cov and mat row mismatch: {cov.shape[0]} vs {n_cells}")

    # design matrix with intercept
    X = np.concatenate([np.ones((n_cells, 1), dtype=float), cov.astype(float)], axis=1)  # (n_cells, k+1)
    # normal eqn: betas = (X^T X)^{-1} X^T Y
    XtX = X.T.dot(X)    # (k+1, k+1)
    # for numerical stability, use lstsq if XtX singular (but normally k small)
    try:
        XtX_inv = np.linalg.inv(XtX)
        betas = XtX_inv.dot(X.T.dot(mat))   # (k+1, n_genes)
    except np.linalg.LinAlgError:
        # fallback to np.linalg.lstsq per-column (slower) but robust
        betas = np.linalg.lstsq(X, mat, rcond=None)[0].T  # returns (k+1, n_genes) after transpose
        betas = betas.T  # ensure shape consistent (k+1, n_genes)
        betas = betas.reshape(-1, mat.shape[1])  # defensive

    fitted = X.dot(betas)   # (n_cells, n_genes)
    resid = mat - fitted
    return resid, betas


#Functions to run. 1st 3 use cycling as a covariate:
#A gene by pseudotime
def gene_vs_pseudotime(adata, query_gene,
                       layer='log1p_norm',
                       detection_thresh=0.01,
                       cluster_key='cluster_annotations',
                       sample_key='sample',
                       covariate='cell_cycle_score',
                       n_bins=40,
                       save_prefix='gene_pt',
                       adata_type='malignant',
                       compute_cycling_if_missing=True):
    """
    Compute/plot correlation of `query_gene` vs dpt_pseudotime and per-cluster/per-sample checks.
    Returns: dict of results (overall rho/p, per-cluster df, per-sample df)
    """
  
    # 1) get expression vector
  
    if layer in adata.layers:
        mat = adata.layers[layer]
        var = np.array(adata.var_names)
        cells = np.array(adata.obs_names)
    else:
        print("Layer not found")
        print(adata)

    if sp.issparse(mat):
        mat = mat.toarray()

    if query_gene not in var:
        raise ValueError(f"{query_gene} not found in var_names")

    out_dir=Path(f"{query_gene}_correlated_with_pt")
    out_dir.mkdir(parents=True, exist_ok=True)


    g_idx = int(np.where(var == query_gene)[0]) #index where query gene is found in adata.obs
    expr = mat[:, g_idx].astype(float) #Getting all the values of the query gene

    # detection check
    detect_frac = (expr > 0).sum() / expr.shape[0]
    if detect_frac < detection_thresh:
        print(f"Warning: {query_gene} detected in {detect_frac:.3%} cells (< {detection_thresh:.0%} threshold). Results may be noisy.")

    # 2) pseudotime and valid cells
    
    if 'dpt_pseudotime' not in adata.obs.columns:
        raise ValueError("dpt_pseudotime not present in adata.obs")

    pt = adata.obs['dpt_pseudotime'].astype(float).values #Getting psuedo time values
    valid_mask = np.isfinite(pt)
    expr = expr[valid_mask]
    pt = pt[valid_mask]
    obs = adata.obs.iloc[np.where(valid_mask)[0]].copy()  # keep index alignment

    
    # 3) overall Spearman
   
    rho_all, p_all = spearmanr(expr, pt, nan_policy='omit') #Correlating the querry gene with pseudotime
    print(f"Overall Spearman: rho={rho_all:.3f}, p={p_all:.2e}")

   
    # 4) scatter + smoothed trend (rolling binned mean) plots
    
    df = pd.DataFrame({'expr': expr, 'pt': pt}, index=obs.index).sort_values('pt')
    # rolling smoothing using bin approach
    window = max(5, int(0.03 * df.shape[0]))  # 3% window
    df['smoothed'] = df['expr'].rolling(window=window, center=True, min_periods=1).mean()

    plt.figure(figsize=(6,4))
    plt.scatter(df['pt'], df['expr'], s=6, alpha=0.25)
    plt.plot(df['pt'], df['smoothed'], color='red', linewidth=2)
    plt.xlabel('dpt_pseudotime'); plt.ylabel(f"{query_gene} (log1p)")
    plt.title(f"{query_gene} vs pseudotime (rho={rho_all:.2f})")
    plt.tight_layout()
    plt.savefig(out_dir/f"{adata_type}_{save_prefix}_{query_gene}_scatter.png", dpi=200)
    plt.close()


    # 5) binned mean ± SEM
    
    df['bin'] = pd.cut(df['pt'], bins=n_bins, labels=False)
    grouped = df.groupby('bin')['expr']
    bin_centers = df.groupby('bin')['pt'].mean()
    means = grouped.mean()
    sems = grouped.sem().fillna(0)

    plt.figure(figsize=(6,3))
    plt.plot(bin_centers, means, marker='o')
    plt.fill_between(bin_centers, means - sems, means + sems, alpha=0.25)
    plt.xlabel('dpt_pseudotime'); plt.ylabel(f"{query_gene} (mean ± SEM)")
    plt.title(f"Binned expression: {query_gene}")
    plt.tight_layout()
    plt.savefig(out_dir/f"{adata_type}_{save_prefix}_{query_gene}_binned.png", dpi=200)
    plt.close()

   
    # 6) per-cluster Spearman
    
    per_cluster = []
    if cluster_key in obs.columns:
        clusters = obs[cluster_key].astype(str).unique()
        for cl in clusters:
            mask = (obs[cluster_key].astype(str) == str(cl)).values
            if mask.sum() < 6:
                per_cluster.append((cl, np.nan, np.nan, mask.sum()))
                continue
            r, p = spearmanr(expr[mask], pt[mask], nan_policy='omit')
            per_cluster.append((cl, float(r), float(p), int(mask.sum())))
        per_cluster_df = pd.DataFrame(per_cluster, columns=['cluster','rho','p','n_cells']).sort_values('rho', ascending=False)
        per_cluster_df.to_csv(f"{save_prefix}_{query_gene}_per_cluster_spearman.csv", index=False)
    else:
        per_cluster_df = pd.DataFrame()

    
    # 7) per-sample Spearman and meta-summary
    
    per_sample = []
    if sample_key in obs.columns:
        samples = obs[sample_key].astype(str).unique()
        for s in samples:
            mask = (obs[sample_key].astype(str) == str(s)).values
            if mask.sum() < 6:
                per_sample.append((s, np.nan, np.nan, mask.sum()))
                continue
            r, p = spearmanr(expr[mask], pt[mask], nan_policy='omit')
            per_sample.append((s, float(r), float(p), int(mask.sum())))
        per_sample_df = pd.DataFrame(per_sample, columns=['sample','rho','p','n_cells']).sort_values('rho', ascending=False)
        per_sample_df.to_csv(f"{save_prefix}_{query_gene}_per_sample_spearman.csv", index=False)
    else:
        per_sample_df = pd.DataFrame()

    
    # 8) partial correlation controlling for a covariate 
    #    We do this by regressing out the covariate from expr and pt, then correlating residuals.
    
    partial_rho, partial_p = (np.nan, np.nan)
    if covariate is not None:
        if covariate not in obs.columns:
            if covariate=="cell_cycle_score" and compute_cycling_if_missing:
                try:
                    compute_cycling_score(adata, layer = layer, score_name="cell_cycle_score")
                    obs[covariate] = adata.obs.loc[obs.index, covariate].astype(float).values
                    print("Computed cycling score and wrote to adata.obs")
                except Exception as e:
                    print(f"Count not compute cycling score:{e}")
            else:
                print("Covariate could not be found in adata.obs; skipping parital correlation")
        if covariate in obs.columns:
            cov = obs[covariate].astype(float).values
            
            # drop NaNs in cov
            valid2 = np.isfinite(cov)
            n_valid_cov=int(valid2.sum())
            if n_valid_cov >= 50:
                try:
                    expr_mat=expr.reshape(-1,1)
                    pt_mat=pt.reshape(-1,1)
                    cov_mat=cov.reshape(-1,1)
                    resid_expr_mat,_=regress_out_covariate_lstsq(expr_mat, cov_mat)
                    resid_pt_mat,_=regress_out_covariate_lstsq(pt_mat, cov_mat)

                    resid_expr=np.asarray(resid_expr_mat.ravel())
                    resid_pt=np.asarray(resid_pt_mat.ravel())
                    mask_valid_resid = np.isfinite(resid_expr) & np.isfinite(resid_pt)
                    n_mask = int(mask_valid_resid.sum())
                    if n_mask >=100:
                        partial_rho,partial_p=spearmanr(resid_expr[mask_valid_resid], resid_pt[mask_valid_resid], nan_policy='omit')
                        print(f"Partial Spearman (controlling for {covariate}): rho={partial_rho:.3f}, p={partial_p:.2e}")
                    else:
                        print(f"Not enought valid residual rows after regression (n={n_mask})")
                except Exception as e:
                    print(f"Failed to compute partial correlation controlling for {covariate}:{e}")
            else:
                print(f"Not enough non-NaN values for covariate {covariate} to compute partial correlation.")
        else:
            print("No covariates requestedl skipping partial correlation")
    #Plotting covariates
    cov_plots_prefix = f"{adata_type}_{save_prefix}_{query_gene}_cov"
    if covariate is not None:
        # check if covariate exits
        if covariate not in obs.columns:
            print(f"Covariate '{covariate}' not in obs — skipping covariate diagnostics.")
        else:
            cov_full = adata.obs[covariate].astype(float).values
            cov = cov_full[valid_mask]  # aligned to expr/pt used above

            # 1) covariate vs pseudotime scatter (are they correlated?)
            plt.figure(figsize=(5,4))
            mask_cp = np.isfinite(cov) & np.isfinite(pt)
            if mask_cp.sum() > 2:
                rp, pp = spearmanr(cov[mask_cp], pt[mask_cp], nan_policy='omit')
                plt.scatter(pt[mask_cp], cov[mask_cp], s=8, alpha=0.3)
                plt.xlabel('dpt_pseudotime'); plt.ylabel(covariate)
                plt.title(f"{covariate} vs pseudotime (rho={rp:.3f}, p={pp:.1e})")
            else:
                plt.text(0.5, 0.5, "Not enough data for cov vs pt", ha='center')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_dir/f"{cov_plots_prefix}_vs_pt.png", dpi=200)
            plt.close()

            # 2) covariate vs query gene expression (does cov explain the gene?)
            plt.figure(figsize=(5,4))
            mask_ce = np.isfinite(cov) & np.isfinite(expr)
            if mask_ce.sum() > 2:
                rq, pq = spearmanr(cov[mask_ce], expr[mask_ce], nan_policy='omit')
                plt.scatter(cov[mask_ce], expr[mask_ce], s=8, alpha=0.3)
                plt.xlabel(covariate); plt.ylabel(f"{query_gene} expr")
                plt.title(f"{covariate} vs {query_gene} (rho={rq:.3f}, p={pq:.1e})")
            else:
                plt.text(0.5, 0.5, "Not enough data for cov vs expr", ha='center')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_dir/f"{cov_plots_prefix}_vs_{query_gene}.png", dpi=200)
            plt.close()

            # 3) per-cluster covariate vs expr (raw) and per-cluster partial (if residuals available)
            per_cluster_cov = []
            clusters = obs[cluster_key].astype(str).unique() if cluster_key in obs.columns else ['all']
            for cl in clusters:
                cmask = (obs[cluster_key].astype(str) == str(cl)).values if cluster_key in obs.columns else np.ones_like(cov, dtype=bool)
                # require some cells
                n = int(cmask.sum())
                if n < 3:
                    per_cluster_cov.append((cl, np.nan, np.nan, n, np.nan, np.nan, 0))
                    continue
                valid_raw_mask = cmask & np.isfinite(cov) & np.isfinite(expr)
                if valid_raw_mask.sum() < 3:
                    per_cluster_cov.append((cl, np.nan, np.nan, n, np.nan, np.nan, int(valid_raw_mask.sum())))
                    continue
                r_raw, p_raw = spearmanr(cov[valid_raw_mask], expr[valid_raw_mask], nan_policy='omit')
                # partial: residualize expr and cov within the cluster if enough pts, else use global residuals if computed
                r_par = np.nan; p_par = np.nan; n_par = 0
                try:
                    # try cluster-local residualization if dpt or cov present
                    # here we remove cov from expr: regress expr ~ cov (within cluster) then correlate residuals with pt residuals if desired
                    # but simpler: regress expr ~ cov within cluster and correlate residual expr vs pt (or compare residuals)
                    # We'll compute residualized expr within-cluster and then compute corr with pt (if available)
                    if np.isfinite(pt).sum() > 0:
                        # resid expr vs resid pt (both within cluster) -- require finite pt in cluster
                        mask_cov = cmask & np.isfinite(cov) & np.isfinite(pt) & np.isfinite(expr)
                        if mask_cov.sum() >= 100:
                            expr_sub = expr[mask_cov]
                            pt_sub = pt[mask_cov]
                            cov_sub = cov[mask_cov]
                            # residualize expr and pt on cov_sub using lstsq helper
                            resid_e_mat, _ = regress_out_covariate_lstsq(expr_sub.reshape(-1,1), cov_sub.reshape(-1,1))
                            resid_p_mat, _ = regress_out_covariate_lstsq(pt_sub.reshape(-1,1), cov_sub.reshape(-1,1))
                            resid_e = resid_e_mat.ravel(); resid_p = resid_p_mat.ravel()
                            r_par, p_par = spearmanr(resid_e, resid_p, nan_policy='omit')
                            n_par = mask_cov.sum()
                except Exception:
                    r_par = np.nan; p_par = np.nan; n_par = 0
                per_cluster_cov.append((cl, float(r_raw), float(p_raw), int(n), float(r_par) if np.isfinite(r_par) else np.nan, float(p_par) if np.isfinite(p_par) else np.nan, int(n_par)))
            pcov_df = pd.DataFrame(per_cluster_cov, columns=['cluster','cov_vs_expr_r','cov_vs_expr_p','n_cells','cov_partial_r','cov_partial_p','n_cells_partial'])
            pcov_df.to_csv(out_dir/f"{cov_plots_prefix}_per_cluster.csv", index=False)

            # simple barplot of raw vs partial cov vs expr per cluster (merge as needed)
            try:
                plt.figure(figsize=(6, max(3, 0.3*pcov_df.shape[0])))
                y = np.arange(pcov_df.shape[0])
                h = 0.35
                plt.barh(y - h/2, pcov_df['cov_vs_expr_r'].fillna(0), height=h, label='raw', alpha=0.9)
                plt.barh(y + h/2, pcov_df['cov_partial_r'].fillna(0), height=h, label=f'partial(−{covariate})', alpha=0.6)
                plt.yticks(y, pcov_df['cluster'].astype(str))
                plt.axvline(0, color='k', lw=0.6)
                plt.xlabel('Spearman rho (covariate vs expr)')
                plt.title(f"{covariate} effect on {query_gene} per cluster")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir/f"{cov_plots_prefix}_per_cluster_bar.png", dpi=200)
                plt.close()
            except Exception as e:
                print("Could not make per-cluster covariate barplot:", e)
    # 9) return results dict

    summary_df = pd.DataFrame([{
    'gene': query_gene,
    'detect_frac': detect_frac,
    'rho_all': float(rho_all),
    'p_all': float(p_all),
    'partial_rho': float(partial_rho) if np.isfinite(partial_rho) else np.nan,
    'partial_p': float(partial_p) if np.isfinite(partial_p) else np.nan,
    }])

    summary_df.to_csv(
        out_dir/f"{query_gene}_gene_vs_pseudotime_summary.csv",
        index=False
    )
    if isinstance(per_cluster_df, pd.DataFrame) and not per_cluster_df.empty:
        pc = per_cluster_df.copy()
        pc['gene'] = query_gene
        pc['rho_all'] = rho_all
        pc['partial_rho_all'] = partial_rho

        pc.to_csv(
            out_dir/f"{query_gene}_gene_vs_pseudotime_per_cluster_with_summary.csv",
            index=False
        )
    if isinstance(per_sample_df, pd.DataFrame) and not per_sample_df.empty:
        ps = per_sample_df.copy()
        ps['gene'] = query_gene
        ps['rho_all'] = rho_all
        ps['partial_rho_all'] = partial_rho

        ps.to_csv(
            out_dir/f"{query_gene}_gene_vs_pseudotime_per_sample.csv",
            index=False
        )

    return None

#All genes vs psuedotime
def allgenes_vs_pseudotime(adata,
                   layer='log1p_norm',
                   detection_thresh=0.01,
                   cluster_key='cluster_annotations',
                   sample_key='sample',
                   covariate='cell_cycle_score',
                   n_top_plot=10,
                   save_prefix='genes_pt',
                   adata_type='malignant',
                   topk_heatmap=40):
    """
    Compute Spearman (or Pearson) correlation of EVERY gene vs dpt_pseudotime using
    your vectorized helpers. Optionally compute partial correlations controlling for
    a covariate by residualizing expression and pseudotime on the covariate.

    Returns results_df with columns:
      ['gene','detect_frac','rho','p','partial_rho','partial_p','cov_rho','cov_p']

    Side-effects: writes several diagnostic plots / csv files to cwd.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.sparse as sp

    # --- 1) get matrix + var names ---
    if layer in getattr(adata, "layers", {}):
        mat = adata.layers[layer]
        var = np.array(adata.var_names)
        cells = np.array(adata.obs_names)
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers")

    # convert sparsity as needed (slice before densify)
    if sp.issparse(mat):
        mat = mat.tocsr()

    # --- 2) pseudotime + valid cells ---
    if 'dpt_pseudotime' not in adata.obs.columns:
        raise ValueError("dpt_pseudotime not present in adata.obs")
    
    out_dir=Path("Allgenes_correlation_with_pt")
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_full = adata.obs['dpt_pseudotime'].astype(float).values
    valid_mask = np.isfinite(pt_full)
    if valid_mask.sum() < 10:
        raise ValueError("Too few valid pseudotime values after filtering NaNs.")
    pt = pt_full[valid_mask]
    obs = adata.obs.iloc[np.where(valid_mask)[0]].copy()

    # slice matrix to valid cells then densify if needed
    if sp.issparse(mat):
        mat = mat[valid_mask, :].toarray()
    else:
        mat = mat[valid_mask, :]

    mat = np.asarray(mat, dtype=float)   # (n_cells, n_genes)
    n_cells, n_genes = mat.shape

    # detection fraction
    detect_frac = (mat > 0).sum(axis=0) / float(n_cells)

    # memory warning
    if n_genes > 50000 or (mat.nbytes / (1024**3)) > 1.0:
        print(f"Warning: large matrix ({n_genes} genes, {n_cells} cells, ~{mat.nbytes/(1024**3):.2f} GB).")

    # --- 3) vectorized Spearman (or Pearson) gene vs pseudotime ---
    print(f"Computing vectorized Spearman (gene vs pseudotime) for {n_genes} genes...")
    rho, pvals = compute_spearman_vectorized(mat, pt)  # returns (rho_array, pvals_array)

    # --- 4) partial correlations: residualize on covariate if requested ---
    partial_rho = np.full(n_genes, np.nan, dtype=float)
    partial_p = np.full(n_genes, np.nan, dtype=float)
    if covariate is not None and covariate in obs.columns:
        cov_full = obs[covariate].astype(float).values
        valid_cov = np.isfinite(cov_full)
        if valid_cov.sum() >= 100:
            try:
                # residualize expression matrix and pt on cov_full (NaN-preserving)
                resid_mat_full, resid_cov = regress_out_covariate_lstsq(mat, cov_full)
                # resid_cov is 1D vector of residualized cov if single cov
                # Now compute vectorized correlations between residualized pt and each residualized gene.
                # Choose rows where resid_pt is finite:
                resid_pt_full, _ = regress_out_covariate_lstsq(pt.reshape(-1,1), cov_full.reshape(-1,1))
                resid_pt = resid_pt_full.ravel()
                valid_rows = np.isfinite(resid_pt)
                n_valid = int(valid_rows.sum())
                if n_valid >= 100:
                    print(f"Computing vectorized Spearman on residuals ({n_valid} cells)...")
                    # subset residuals to valid rows
                    sub_resid_mat = resid_mat_full[valid_rows, :]
                    sub_resid_pt = resid_pt[valid_rows]
                    prho, pp = compute_spearman_vectorized(sub_resid_mat, sub_resid_pt)
                    partial_rho[:] = prho
                    partial_p[:] = pp
                else:
                    print(f"Not enough valid rows after residualization ({n_valid}) to compute partial correlations.")
            except Exception as e:
                print(f"Failed to compute partial correlations by residualization: {e}")
        else:
            print(f"Not enough finite covariate values ({valid_cov.sum()}) to compute partial correlations.")
    else:
        if covariate is not None:
            print(f"Covariate '{covariate}' not found in obs; skipping partial correlations.")

    # --- 5) covariate effects: compute cov_rho (gene vs cov) and residual matrix if possible ---
    cov_rho = np.full(n_genes, np.nan, dtype=float)
    cov_p = np.full(n_genes, np.nan, dtype=float)
    resid_mat_for_plot = None
    if covariate is not None and covariate in obs.columns:
        covvals = obs[covariate].astype(float).values
        valid_cov = np.isfinite(covvals)
        if valid_cov.sum() >= 100:
            try:
                # compute vectorized Spearman between cov and genes on rows with finite cov
                cov_sub = covvals[valid_cov]
                mat_sub = mat[valid_cov, :]
                cov_rho, cov_p = compute_spearman_vectorized(mat_sub, cov_sub)
            except Exception as e:
                print("Vectorized cov vs gene failed, falling back to loop:", e)
                for gi in range(n_genes):
                    mask = valid_cov & np.isfinite(mat[:, gi])
                    if mask.sum() >= 100:
                        rtmp, ptmp = spearmanr(mat[mask, gi], covvals[mask], nan_policy='omit')
                        cov_rho[gi] = rtmp; cov_p[gi] = ptmp
            # try compute residualized matrix (for plotting partial heatmaps)
            try:
                resid_mat_for_plot, _ = regress_out_covariate_lstsq(mat, covvals)
            except Exception as e:
                resid_mat_for_plot = None
                print("Could not compute residualized matrix for plotting:", e)
        else:
            print(f"Not enough finite covariate values ({valid_cov.sum()}) for cov diagnostics.")
    # else: cov absent -> skipped

    # --- 6) assemble results DataFrame ---
    results_df = pd.DataFrame({
        'gene': var,
        'detect_frac': detect_frac,
        'rho': rho,
        'p': pvals,
        'partial_rho': partial_rho,
        'partial_p': partial_p,
        'cov_rho': cov_rho,
        'cov_p': cov_p
    })
    results_df = results_df.sort_values('rho', ascending=False).reset_index(drop=True)

    # save csv
    out_csv = out_dir/f"{adata_type}_{save_prefix}_all_genes_spearman.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved all-gene spearman results to {out_csv}")

    #  plotting diagnostics (cov_rho vs rho scatter, top drivers, residual heatmap) 
    cov_plots_prefix = f"{adata_type}_{save_prefix}_cov_effects"
    # scatter cov_rho vs rho
    if np.isfinite(cov_rho).any():
        plt.figure(figsize=(5,5))
        plt.scatter(results_df['rho'], results_df['cov_rho'], s=6, alpha=0.5)
        plt.axhline(0, color='gray', lw=0.6); plt.axvline(0, color='gray', lw=0.6)
        plt.xlabel('rho (gene vs pseudotime)')
        plt.ylabel(f'rho (gene vs {covariate})')
        plt.title(f"Covariate effect: {covariate} vs pseudotime correlations")
        out = out_dir/f"{cov_plots_prefix}_rho_vs_covrho.png"
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
        print("Saved cov vs rho scatter to", out)

        # annotate / top drivers by abs(rho)*abs(cov_rho)
        results_df['driver_score'] = np.abs(results_df['rho']) * np.abs(results_df['cov_rho'])
        top_drivers = results_df.dropna(subset=['driver_score']).sort_values('driver_score', ascending=False).head(10)
        if top_drivers.shape[0] > 0:
            plt.figure(figsize=(6, max(3, 0.25 * top_drivers.shape[0])))
            sns.barplot(x='driver_score', y='gene', data=top_drivers, palette='magma')
            plt.title('Top genes likely driven by covariate (|rho|*|cov_rho|)')
            out = out_dir/f"{cov_plots_prefix}_top_drivers.png"
            plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
            print("Saved covariate-driven gene barplot to", out)

    # residual heatmap for top cov genes (if residuals available)
    if resid_mat_for_plot is not None and np.isfinite(cov_rho).any():
        df_cov = results_df.dropna(subset=['cov_rho']).copy()
        df_cov['abs_cov_rho'] = df_cov['cov_rho'].abs()
        topk = min(topk_heatmap, df_cov.shape[0])
        top_cov_genes = df_cov.sort_values('abs_cov_rho', ascending=False).head(topk)['gene'].tolist()
        if len(top_cov_genes) > 0:
            idxs = [int(np.where(var == g)[0]) for g in top_cov_genes]
            heat = resid_mat_for_plot[:, idxs]  # (n_cells x k)
            # sample cells if too many; prefer pseudotime ordering if available
            nplot = min(2000, heat.shape[0])
            if heat.shape[0] > nplot and 'dpt_pseudotime' in adata.obs.columns:
                order = np.argsort(adata.obs['dpt_pseudotime'].astype(float).values)
                sel = np.linspace(0, len(order)-1, nplot).astype(int)
                rows = order[sel]
            else:
                rows = np.arange(heat.shape[0])[:nplot]
            hplot = heat[rows, :]
            h_z = (hplot - np.nanmean(hplot, axis=0)) / np.nanstd(hplot, axis=0)
            plt.figure(figsize=(max(6, 0.25 * len(top_cov_genes)), 6))
            sns.heatmap(h_z.T, yticklabels=top_cov_genes, xticklabels=False, cmap='vlag', cbar_kws={'label': 'z-score'})
            plt.title(f"Top {len(top_cov_genes)} genes by |cov_rho| (resid expr)")
            out = out_dir/f"{cov_plots_prefix}_top{len(top_cov_genes)}_covheat.png"
            plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
            print("Saved covariate residual heatmap to", out)

    #plots for top genes by rho (binned mean along pseudotime) 
    if n_top_plot and n_top_plot > 0:
        top_pos = results_df.head(n_top_plot)['gene'].tolist()
        top_neg = results_df.tail(n_top_plot)['gene'].tolist()[::-1]
        genes_to_plot = top_pos + top_neg
        nplot = len(genes_to_plot)
        fig, axes = plt.subplots(nplot, 1, figsize=(6, 2.2 * nplot), squeeze=False)
        for i, g in enumerate(genes_to_plot):
            g_idx = int(np.where(var == g)[0])
            gexpr = mat[:, g_idx]
            dfg = pd.DataFrame({'expr': gexpr, 'pt': pt})
            dfg = dfg.sort_values('pt')
            dfg['bin'] = pd.cut(dfg['pt'], bins=50, labels=False)
            grouped = dfg.groupby('bin')['expr']
            bin_centers = dfg.groupby('bin')['pt'].mean()
            means = grouped.mean().fillna(0)
            sems = grouped.sem().fillna(0)
            ax = axes[i, 0]
            ax.plot(bin_centers, means)
            ax.fill_between(bin_centers, means - sems, means + sems, alpha=0.25)
            ax.set_ylabel(g)
            if i == nplot - 1:
                ax.set_xlabel('dpt_pseudotime')
        plt.tight_layout()
        plotfile = out_dir/f"{adata_type}_{save_prefix}_top_genes_binned.png"
        plt.savefig(plotfile, dpi=200)
        plt.close()
        print(f"Saved top genes plot to {plotfile}")

    return results_df

#A gene vs all genes
def coexpression_of_query_with_all_genes(adata,
         query_gene="ZNF423",
         layer = "log1p_norm",
         method = "spearman",
         covariate= "cell_cycle_score",
         out_prefix= "ZNF423_coexpr",
         topn=12,
         topk_heatmap=40,
         detection_thresh=0.01,
         compute_cycling_if_missing=True):

    if layer not in adata.layers:
        raise ValueError(f"Layer {layer} not found in adata.layers")
    else:
        mat=adata.layers[layer]
    

    # handle sparse
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=float)  # (n_cells, n_genes)
    var = np.array(adata.var_names)

    #Extract querry gene
    if query_gene not in var:
        raise ValueError(f"Query gene '{query_gene}' not found in var_names.")
    
    out_dir=Path(f"{query_gene}_correlation_with_allgenes")
    out_dir.mkdir(parents=True, exist_ok=True)


    q_idx = int(np.where(var == query_gene)[0])
    q_expr = mat[:, q_idx]
    n_cells = mat.shape[0]

    # detection fraction for each gene
    detect_frac = (mat > 0).sum(axis=0) / float(n_cells)

    # optional check for detection of query
    q_detect_frac = detect_frac[q_idx]
    if q_detect_frac < detection_thresh:
        print(f"Warning: query gene {query_gene} detected in {q_detect_frac:.3%} cells (< {detection_thresh:.0%})")

    # compute correlations
    print(f"Computing {method} correlations ({query_gene} vs {mat.shape[1]} genes)...")
    if method == 'spearman':
        rho, pvals = compute_spearman_vectorized(mat, q_expr)
    elif method == 'pearson':
        rho, pvals = compute_pearson_vectorized(mat, q_expr)
    else:
        raise ValueError("Unknown method: choose 'spearman' or 'pearson'.")


    partial_rho = np.full_like(rho, np.nan, dtype=float)
    partial_p = np.full_like(pvals, np.nan, dtype=float)

    if covariate is not None:
        # ensure covariate exists or compute if allowed
        if covariate not in adata.obs.columns:
            if covariate == "cell_cycle_score" and compute_cycling_if_missing:
                try:
                    compute_cycling_score(adata, layer=layer, score_name="cell_cycle_score")
                    print("Computed cycling_score and wrote to adata.obs")
                except Exception as e:
                    print("Could not compute cycling score:", e)
            else:
                print("Covariate not found; skipping partial correlation.")
        if covariate in adata.obs.columns:
            cov_full = adata.obs[covariate].astype(float).values  # aligned to full adata
            valid_cov = np.isfinite(cov_full)
            if valid_cov.sum() >= 10:
                # restrict to rows with finite cov AND finite pt (we already restricted to valid pt earlier)
                # mat is already subset to valid pseudotime rows earlier in your function,
                # so cov_sub must be aligned to mat's rows (mat is the filtered mat)
                cov_sub = cov_full[valid_cov]
                mat_sub = mat[valid_cov, :]  # (n_cov_cells, n_genes)
                # residualize matrix on cov_sub; returns resid_mat_full shape (n_cov_cells, n_genes)
                try:
                    resid_mat_full, _ = residualize_matrix_on_covariate(mat_sub, cov_sub)
                except Exception as e:
                    resid_mat_full = None
                    print("Residualizing full matrix failed:", e)

                # residualize query gene (vector) on cov_sub
                try:
                    q_mat = q_expr[valid_cov].reshape(-1, 1)
                    resid_q_mat, _ = residualize_matrix_on_covariate(q_mat, cov_sub)
                    resid_q = resid_q_mat.ravel()
                except Exception as e:
                    resid_q = None
                    print("Residualizing query gene failed:", e)

                # compute vectorized Spearman between resid_q and each resid gene
                if (resid_mat_full is not None) and (resid_q is not None):
                    # ensure shapes
                    resid_mat_full = np.asarray(resid_mat_full)
                    if resid_mat_full.ndim == 1:
                        resid_mat_full = resid_mat_full.reshape(-1, 1)
                    # restrict to rows where both residuals finite
                    valid_rows = np.isfinite(resid_q) & np.all(np.isfinite(resid_mat_full), axis=1)
                    n_valid_rows = int(valid_rows.sum())
                    if n_valid_rows >= 10:
                        try:
                            prho, pp = compute_spearman_vectorized(resid_mat_full[valid_rows, :], resid_q[valid_rows])
                            partial_rho[:] = prho
                            partial_p[:] = pp
                            print(f"Computed partial correlations on {n_valid_rows} cells (residualized by {covariate})")
                        except Exception as e:
                            print("compute_spearman_vectorized failed on residuals:", e)
                    else:
                        print(f"Not enough valid rows after residualization ({n_valid_rows}); skipping partials.")
                else:
                    print("Residual matrices not available; skipping partial correlations.")
            else:
                print(f"Not enough finite covariate values ({valid_cov.sum()}) to compute partial correlations.")

    # assemble results DataFrame
    df = pd.DataFrame({
        'gene': var,
        'detect_frac': detect_frac,
        'rho': rho,
        'p': pvals,
        'partial_rho': partial_rho,
        'partial_p': partial_p
    })

    # FDR
    df['fdr'] = benjamini_hochberg(df['p'].values)

    # sort by rho descending
    df = df.sort_values('rho', ascending=False).reset_index(drop=True)

    out_csv = out_dir/f"{out_prefix}_coexpression_{query_gene}.csv"
    df.to_csv(out_csv, index=False)
    print("Saved results to", out_csv)

    df_plot_base=df[df['gene']!= query_gene].copy()

    if df['partial_rho'].notna().any():
        plt.figure(figsize=(5,5))
        plt.axhline(0, color='gray', linewidth = 0.6)
        plt.axvline(0, color='gray', linewidth = 0.6)
        plt.scatter(df['rho'], df['partial_rho'], s=6, alpha=0.6)
        plt.xlabel("Raw Rho")
        plt.ylabel("Partial Rho")
        plt.title(f"Raw vs Partial correlations ({query_gene})")

        df_nonan=df_plot_base.dropna(subset=['rho','partial_rho']).copy()
        df_nonan['delta']=df_nonan['partial_rho']-df_nonan['rho']
        top_inc=df_nonan.sort_values('delta',ascending=False).head(5)
        top_dec=df_nonan.sort_values('delta').head(5)
        for _,row in pd.concat([top_inc,top_dec]).iterrows():
            plt.text(row['rho'],row['partial_rho'], row['gene'], fontsize=8)

        out=out_dir/f"{query_gene}_raw_vs_partial_scatter.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved raw vs partial scatter")
    else:
        print("No partial_rho found - skipping scatter plot")
    
    if df['partial_rho'].notna().any():
        plt.figure(figsize=(5,3))
        sns.histplot(df_plot_base['partial_rho'].dropna(), kde=True, bins=40)
        plt.xlabel('Partial rho')
        plt.title(f"Distribution of partial correlations vs {query_gene}")
        out = out_dir/f"{query_gene}_partial_rho_hist.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved partial rho histogram")

    # diagnostics / figures
    if resid_mat_full is not None and topn and topn > 0:
        df_noquery=df_plot_base.copy()
        # exclude query gene itself from plotting (it will be present at rho=1)

        df_noquery = df_noquery.dropna(subset=['partial_rho'])
        top_pos = df_noquery.sort_values('partial_rho',ascending=False).head(topn)['gene'].tolist()
        top_neg = df_noquery.sort_values('partial_rho',ascending=True).head(topn)['gene'].tolist()

        # scatter grid for top positive and negative genes
        nplots = len(top_pos) + len(top_neg)
        if nplots > 0:
            ncols = min(3, nplots)
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
            axes = np.array(axes).reshape(-1)
            genes_to_plot = top_pos + top_neg
            for ax, g in zip(axes, genes_to_plot):
                gi = int(np.where(var == g)[0])
                x=resid_q
                y=resid_mat_full[:,gi]
                mask_xy=np.isfinite(x)&np.isfinite(y)
                ax.scatter(x[mask_xy], y[mask_xy], s=6, alpha=0.25)
                ax.set_xlabel(f"{query_gene} (resid)")
                ax.set_ylabel(g + " (resid)")
                r = df.loc[df['gene'] == g, 'partial_rho'].values[0]
                ax.set_title(f"r={r:.2f}")
            # hide unused axes
            for ax in axes[len(genes_to_plot):]:
                ax.axis('off')
            plt.tight_layout()
            scatter_file = out_dir/f"{out_prefix}_{query_gene}_top{topn}_scatter.png"
            plt.savefig(scatter_file, dpi=300)
            plt.close()
            print("Saved scatter grid to", scatter_file)
    else:
        print("Residuals not available or topn=0, skilling paritail residual")
    topk = topk_heatmap
    if resid_mat_full is not None and topk and topk > 0:
        df_noquery = df_plot_base.copy()
        df_noquery['abs_partial'] = df_noquery['partial_rho'].abs()
        topk_genes = df_noquery.sort_values('abs_partial', ascending=False).head(topk)['gene'].tolist()
        if len(topk_genes) > 0:
            idxs = [int(np.where(var == g)[0]) for g in topk_genes]
            heatmat = resid_mat_full[:, idxs]  # residualized expression (cells x topk)
            # sample cells if too many, prefer pseudotime ordering if available
            nplotcells = min(2000, heatmat.shape[0])
            if heatmat.shape[0] > nplotcells:
                if 'dpt_pseudotime' in adata.obs.columns:
                    pt_all = adata.obs['dpt_pseudotime'].astype(float).values
                    order = np.argsort(pt_all)
                    sel = np.linspace(0, len(order)-1, nplotcells).astype(int)
                    sample_idx = order[sel]
                else:
                    sample_idx = np.random.choice(heatmat.shape[0], size=nplotcells, replace=False)
                heatmat_plot = heatmat[sample_idx, :]
            else:
                heatmat_plot = heatmat
            # z-score genes (columns)
            heat_z = (heatmat_plot - np.nanmean(heatmat_plot, axis=0)) / np.nanstd(heatmat_plot, axis=0)
            plt.figure(figsize=(max(6, len(topk_genes)*0.25), 6))
            sns.heatmap(heat_z.T, xticklabels=False, yticklabels=topk_genes, cmap='vlag', cbar_kws={'label': 'z-score'})
            plt.xlabel('cells (sampled)')
            plt.title(f"Top {len(topk_genes)} genes by |partial rho| with {query_gene} (resid)")
            out = out_dir/f"{out_prefix}_{query_gene}_top{len(topk_genes)}_partial_heatmap.png"
            plt.tight_layout()
            plt.savefig(out, dpi=300)
            plt.close()
            print("Saved partial heatmap to", out)
    else:
        print("Residualized matrix not available or topk <= 0 — skipping partial heatmap.")
    # heatmap of top-k genes (by absolute rho)
    topk = topk_heatmap
    if topk and topk > 0:
        # pick top k by absolute rho (excluding the query)
        df_noquery = df[df['gene'] != query_gene].copy()
        df_noquery['absrho'] = df_noquery['rho'].abs()
        topk_genes = df_noquery.sort_values('absrho', ascending=False).head(topk)['gene'].tolist()
        if len(topk_genes) > 0:
            idxs = [int(np.where(var == g)[0]) for g in topk_genes]
            heatmat = mat[:, idxs]  # (n_cells, topk)
            # Optionally sample cells if huge
            nplotcells = min(2000, heatmat.shape[0])
            if heatmat.shape[0] > nplotcells:
                # sample uniformly across pseudotime if available, otherwise random
                if 'dpt_pseudotime' in adata.obs.columns:
                    pt = adata.obs['dpt_pseudotime'].astype(float).values
                    order = np.argsort(pt)
                    sel = np.linspace(0, len(order)-1, nplotcells).astype(int)
                    sample_idx = order[sel]
                else:
                    sample_idx = np.random.choice(heatmat.shape[0], size=nplotcells, replace=False)
                heatmat_plot = heatmat[sample_idx, :]
            else:
                heatmat_plot = heatmat
            # z-score genes (columns)
            heat_z = (heatmat_plot - np.nanmean(heatmat_plot, axis=0)) / np.nanstd(heatmat_plot, axis=0)
            plt.figure(figsize=(max(6, len(topk_genes)*0.25), 6))
            sns.heatmap(heat_z.T, xticklabels=False, yticklabels=topk_genes, cmap='vlag', cbar_kws={'label': 'z-score'})
            plt.xlabel('cells (sampled)')
            plt.title(f"Top {len(topk_genes)} coexpressed genes with {query_gene}")
            heat_file = out_dir/f"{out_prefix}_{query_gene}_top{len(topk_genes)}_heatmap.png"
            plt.tight_layout()
            plt.savefig(heat_file, dpi=300)
            plt.close()
            print("Saved heatmap to", heat_file)

    print("Done.")

#2 genes correlated. Uses pseudotime as a covariate
def correlate_and_plot_two_genes(
    adata,
    gene1,
    gene2,
    layer='log1p_norm',           # which expression layer to use; fallback to adata.X/raw if missing
    pt_key='dpt_pseudotime',      # pseudotime key used for partial correlation
    cluster_key='cluster_annotations',  # cluster column for coloring / per-cluster stats
    method='spearman',            # 'spearman' or 'pearson'
    min_cells_per_cluster=8,      # skip clusters with fewer cells for per-cluster stats
    figsize=(14,4),
    cmap='tab20',
    scatter_alpha=0.4,
    scatter_s=10,
    per_cluster_partial_mode="global"
):
    """
    Compute correlations between gene1 and gene2 and visualize:
      - left: raw scatter (gene1 vs gene2) colored by cluster
      - middle: scatter of residuals after regressing out pseudotime (partial correlation)
      - right: barplot of per-cluster correlations (raw)
    Returns: results dict and (fig, axes)
    """
    # ---- get expression matrix and gene indices ----
    if layer in getattr(adata, "layers", {}):
        mat = adata.layers[layer] #making matrix of counts
        var_names = np.array(adata.var_names) #Extracting all gene names in adata
    else:
        print(f"{layer} is not in adata.obs")

    # densify if sparse
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=float)

    if gene1 not in var_names:
        raise ValueError(f"{gene1} not found in var_names")
    if gene2 not in var_names:
        raise ValueError(f"{gene2} not found in var_names")

    out_dir=Path(f"{gene1}_{gene2}_correlation")
    out_dir.mkdir(parents=True, exist_ok=True)

    i1 = int(np.where(var_names == gene1)[0][0]) #extracting index for gene 1
    i2 = int(np.where(var_names == gene2)[0][0]) #extracting index for gene 1

    g1 = mat[:, i1].astype(float)
    g2 = mat[:, i2].astype(float)

    n_cells = mat.shape[0]

    #Checking that pseudotime exists
    if pt_key not in adata.obs.columns:
        print(f"Warning: pseudotime key '{pt_key}' not found in adata.obs — partial correlation will be skipped.")
        pt = None
    else:
        pt = adata.obs[pt_key].astype(float).values

    # Making sure cluster key exists
    if cluster_key in adata.obs.columns:
        clusters_all = adata.obs[cluster_key].astype(str).values
    else:
        print("Wrong cluster key")

    # extracting only finite values for both gene 1 and gene 2
    valid_raw = np.isfinite(g1) & np.isfinite(g2)
    if valid_raw.sum() == 0:
        raise ValueError("No overlapping finite expression values between the two genes.")

    # Correlation
    if method == 'spearman':
        overall_rho, overall_p = spearmanr(g1[valid_raw], g2[valid_raw], nan_policy='omit')
    else:
        overall_rho, overall_p = pearsonr(g1[valid_raw], g2[valid_raw])

    # partial correlation - residualize both genes on pseudotime
    partial_rho = np.nan
    partial_p = np.nan
    resid_g1 = None
    resid_g2 = None
    if pt is not None:
        # create full-length arrays and call regress_out_covariate_lstsq
        try:
            cov = pt  # shape (n_cells,)
            expr_pair = np.column_stack([g1, g2]) #g1 and g2 columns
            resid_mat, resid_cov = regress_out_covariate_lstsq(expr_pair, cov)
            # resid_mat has NaNs where cov invalid; extract residual vectors
            resid_g1 = resid_mat[:, 0]
            resid_g2 = resid_mat[:, 1]
            # compute partial correlation on rows where both residuals finite
            valid_part = np.isfinite(resid_g1) & np.isfinite(resid_g2)
            if valid_part.sum() >= 100:
                if method == 'spearman':
                    partial_rho, partial_p = spearmanr(resid_g1[valid_part], resid_g2[valid_part], nan_policy='omit')
                else:
                    partial_rho, partial_p = pearsonr(resid_g1[valid_part], resid_g2[valid_part])
            else:
                print(f"Not enough valid rows ({valid_part.sum()}) to compute partial correlation.")
        except Exception as e:
            print(f"Partial correlation (regressing out '{pt_key}') failed: {e}")
            resid_g1 = None
            resid_g2 = None

    clusters_unique = np.unique(clusters_all)
    per_cluster = []
    per_cluster_partial = []

    for cl in clusters_unique:
        mask = (clusters_all == cl)
        n = int(mask.sum())

        # Raw correlations
        if n < min_cells_per_cluster:
            per_cluster.append((cl, np.nan, np.nan, n))
        else:
            valid = mask & valid_raw
            if valid.sum() < min_cells_per_cluster:
                per_cluster.append((cl, np.nan, np.nan, int(valid.sum())))
            else:
                if method == 'spearman':
                    r, p = spearmanr(g1[valid], g2[valid], nan_policy='omit')
                else:
                    r, p = pearsonr(g1[valid], g2[valid])
                per_cluster.append((cl, float(r), float(p), int(valid.sum())))

        # Partial (global residuals) or per-cluster residualization
        # Keep result aligned with clusters (append one entry per cluster)
        if per_cluster_partial_mode == "global" and (resid_g1 is not None) and (resid_g2 is not None):
            mask2 = mask & np.isfinite(resid_g1) & np.isfinite(resid_g2)
            n_mask2 = int(mask2.sum())
            if n_mask2 < min_cells_per_cluster:
                per_cluster_partial.append((cl, np.nan, np.nan, n_mask2))
            else:
                if method == 'spearman':
                    parr, parp = spearmanr(resid_g1[mask2], resid_g2[mask2], nan_policy='omit')
                else:
                    parr, parp = pearsonr(resid_g1[mask2], resid_g2[mask2])
                per_cluster_partial.append((cl, float(parr), float(parp), n_mask2))

        elif per_cluster_partial_mode == "per_cluster":
            # do a cluster-local residualization (regress both genes on pt within cluster)
            mask_cov = mask & np.isfinite(pt) & np.isfinite(g1) & np.isfinite(g2)
            n_mask_cov = int(mask_cov.sum())
            if n_mask_cov < min_cells_per_cluster:
                per_cluster_partial.append((cl, np.nan, np.nan, n_mask_cov))
            else:
                try:
                    expr_sub = np.column_stack([g1[mask_cov], g2[mask_cov]])  # shape (n_mask_cov, 2)
                    pt_sub = pt[mask_cov]
                    # use your robust residualizer (or regress_out_covariate_lstsq)
                    resid_sub, _ = regress_out_covariate_lstsq(expr_sub, pt_sub)
                    # ensure shape is (n_mask_cov, 2)
                    resid_sub = np.asarray(resid_sub)
                    if resid_sub.ndim == 1:
                        resid_sub = resid_sub.reshape(-1, 2)  # defensive; if shape weird, will raise later
                    valid_sub = np.isfinite(resid_sub[:, 0]) & np.isfinite(resid_sub[:, 1])
                    n_valid_sub = int(valid_sub.sum())
                    if n_valid_sub < min_cells_per_cluster:
                        per_cluster_partial.append((cl, np.nan, np.nan, n_valid_sub))
                    else:
                        if method == 'spearman':
                            parr, parp = spearmanr(resid_sub[valid_sub, 0], resid_sub[valid_sub, 1], nan_policy='omit')
                        else:
                            parr, parp = pearsonr(resid_sub[valid_sub, 0], resid_sub[valid_sub, 1])
                        per_cluster_partial.append((cl, float(parr), float(parp), n_valid_sub))
                except Exception as e:
                    per_cluster_partial.append((cl, np.nan, np.nan, 0))
                    print(f"Per-cluster residualization failed for cluster {cl}: {e}")
        else:
            # unknown mode or no partialization requested
            per_cluster_partial.append((cl, np.nan, np.nan, 0))

    # Convert lists to DataFrames
    per_cluster_df = pd.DataFrame(per_cluster, columns=['cluster', 'rho', 'p', 'n_cells'])
    per_cluster_partial_df = pd.DataFrame(per_cluster_partial, columns=['cluster', 'partial_rho', 'partial_p', 'n_cells_partial'])

    # Sort raw per-cluster df (preserve for plotting)
    per_cluster_df = per_cluster_df.sort_values('rho', ascending=False).reset_index(drop=True)

    # Reset index for partial df too (so we can merge on 'cluster')
    per_cluster_partial_df = per_cluster_partial_df.reset_index(drop=True)

    # ---------- plotting ----------
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1,1,0.7]})

    # Color map for clusters
    unique_clusters = list(per_cluster_df['cluster'].values)
    present_clusters = [c for c in np.unique(clusters_all) if c in unique_clusters]
    cmap_obj = plt.get_cmap(cmap)
    colors = {c: cmap_obj(i % cmap_obj.N) for i, c in enumerate(present_clusters)}

    # Panel 1: raw scatter colored by cluster
    ax = axes[0]
    for c in np.unique(clusters_all):
        mask = clusters_all == c
        if mask.sum() == 0:
            continue
        ax.scatter(g1[mask], g2[mask], s=scatter_s, alpha=scatter_alpha,
                color=colors.get(c, 'gray'), label=str(c), rasterized=True)
    ax.set_xlabel(f"{gene1} expr")
    ax.set_ylabel(f"{gene2} expr")
    ax.set_title(f"Raw: {gene1} vs {gene2}\n{method.title()} rho={overall_rho:.3f}, p={overall_p:.2g}")
    ax.legend(fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')

    # Panel 2: residual scatter (if available)
    ax = axes[1]
    if (resid_g1 is None) or (resid_g2 is None):
        ax.text(0.5, 0.5, f"Partial correlation\n('{pt_key}' missing or failed)", ha='center', va='center')
        ax.set_axis_off()
    else:
        for c in np.unique(clusters_all):
            mask = (clusters_all == c)
            mask2 = mask & np.isfinite(resid_g1) & np.isfinite(resid_g2)
            if mask2.sum() == 0:
                continue
            ax.scatter(resid_g1[mask2], resid_g2[mask2], s=scatter_s, alpha=scatter_alpha,
                    color=colors.get(c, 'gray'), label=str(c), rasterized=True)
        ax.set_xlabel(f"{gene1} residual (wrt {pt_key})")
        ax.set_ylabel(f"{gene2} residual (wrt {pt_key})")
        title = f"Residuals (regressed out {pt_key})\n"
        if np.isfinite(partial_rho):
            title += f"{method.title()} rho={partial_rho:.3f}, p={partial_p:.2g}"
        else:
            title += "not enough data"
        ax.set_title(title)
        ax.legend(fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')

    # Panel 3: per-cluster barplot of raw vs partial correlations
    ax = axes[2]
    # merge raw and partial dataframes on cluster
    pc = per_cluster_df.merge(per_cluster_partial_df, on="cluster", how="left")

    # if no data, render message
    if pc.dropna(subset=['rho', 'partial_rho'], how='all').shape[0] == 0:
        ax.text(0.5, 0.5, "No cluster-level stats (too few cells)", ha='center', va='center')
        ax.set_axis_off()
    else:
        pc_plot = pc.copy()
        # create y positions
        y = np.arange(pc_plot.shape[0])
        h = 0.35
        # colors per cluster (matching list order)
        cluster_colors = [colors.get(c, 'gray') for c in pc_plot['cluster'].astype(str)]
        ax.barh(y - h/2, pc_plot['rho'].fillna(0), height=h, color=cluster_colors, alpha=0.9, label='Raw')
        ax.barh(y + h/2, pc_plot['partial_rho'].fillna(0), height=h, color=cluster_colors, alpha=0.6, label=f'Partial: {pt_key}')
        ax.set_yticks(y)
        ax.set_yticklabels(pc_plot['cluster'].astype(str))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{method.title()} rho")
        ax.set_title("Per-cluster correlations")
        ax.invert_yaxis()
        ax.legend(fontsize='small')

    plt.tight_layout()
    out_fig = out_dir / f"{gene1}_{gene2}_correlation.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    print("Saved plot to:", out_fig)

    # ---------- results dict + save CSVs ----------
    results = {
        'gene1': gene1,
        'gene2': gene2,
        'method': method,
        'overall_r': overall_rho,
        'overall_p': overall_p,
        'partial_r': partial_rho,
        'partial_p': partial_p,
        'per_cluster_df': per_cluster_df,
        'fig': fig,
        'axes': axes,
    }

    summary_keys = ['gene1', 'gene2', 'method', 'overall_r', 'overall_p', 'partial_r', 'partial_p']
    summary = {k: results.get(k, pd.NA) for k in summary_keys}
    summary_df = pd.DataFrame([summary])
    summary_csv = out_dir / f"{results['gene1']}_{results['gene2']}_correlation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary:", summary_csv)

    # save per-cluster CSV if available
    if isinstance(per_cluster_df, pd.DataFrame) and not per_cluster_df.empty:
        per_cluster_csv = out_dir / f"{results['gene1']}_{results['gene2']}_per_cluster.csv"
        per_cluster_df.to_csv(per_cluster_csv, index=False)
        print("Saved per-cluster results:", per_cluster_csv)

    if isinstance(per_cluster_partial_df, pd.DataFrame) and not per_cluster_partial_df.empty:
        per_cluster_partial_csv = out_dir / f"{results['gene1']}_{results['gene2']}_per_cluster_partial.csv"
        per_cluster_partial_df.to_csv(per_cluster_partial_csv, index=False)
        print("Saved per-cluster partial results:", per_cluster_partial_csv)

    return results



