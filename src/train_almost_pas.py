def plot_spectral_biclustering_heatmap(
    relevance_scores,
    cluster_labels,
    feature_names,
    omics_splits,
    output_path,
    omics_colors=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    import os

    # --- Reduce 2048D to 64D using max-over-16 logic ---
    relevance_scores = extract_summary_features_np(relevance_scores)

    if omics_colors is None:
        omics_colors = {
            'mf': '#D62728',     # red
            'cna': '#1F77B4',    # blue
            'meth': '#2CA02C',   # green
            'ge': '#9467BD'      # purple
        }

    cancer_names = [
        'Breast', 'Kidney', 'Rectum', 'Prostate', 'Stomach', 'HeadNeck', 'LungAd', 'Thyroid',
        'Bladder', 'Esophagus', 'Liver', 'Uterus', 'Colon', 'LungSc', 'Cervix', 'KidneyPap'
    ]
    omics_order = ['mf', 'cna', 'ge', 'meth']

    pretty_labels = []
    for omics in omics_order:
        pretty_labels.extend([cancer for cancer in cancer_names])

    # ---- 🔹 Sort nodes by cluster ----
    sorted_indices = np.argsort(cluster_labels)
    sorted_scores = relevance_scores[sorted_indices]
    sorted_clusters = cluster_labels[sorted_indices]

    # ---- 🔹 Feature colors ----
    feature_colors = []
    for i, name in enumerate(feature_names):
        for omics, (start, end) in omics_splits.items():
            if start <= i <= end:
                feature_colors.append(omics_colors[omics])
                break
        else:
            feature_colors.append("#AAAAAA")


    # fig = plt.figure(figsize=(16, 12))
    # gs = fig.add_gridspec(nrows=10, ncols=1)

    fig = plt.figure(figsize=(18, 16))

    gs = fig.add_gridspec(
        nrows=12,  # Increased to add space between heatmap and legend
        ncols=40,
        width_ratios=[
            0.02, *[1]*28, *[0.3]*2, 0.2, 0.1, *[0.02]*7  # same as before
        ],
        height_ratios=[
            *[1]*10,  # rows 0–9: plots
            0.3,      # row 10: spacer (gap between plot and legend)
            0.6       # row 11: legend
        ]
    )


    ax = fig.add_subplot(gs[0:10, 1:29])        # heatmap
    ax_curve = fig.add_subplot(gs[0:10, 29:31], sharey=ax)  # curve
    ax_cbar = fig.add_subplot(gs[0:10, 32])     # colorbar
    ax_legend = fig.add_subplot(gs[11, 1:29])   # legend (bottom-most row)


    ##ax = fig.add_subplot(gs[:-1, 0])       # heatmap
    ##ax_legend = fig.add_subplot(gs[-1, 0]) # legend

    # ✅ Compute cluster boundaries and counts
    unique_clusters, counts = np.unique(sorted_clusters, return_counts=True)
    cluster_boundaries = np.cumsum(counts)
    cluster_start_indices = [0] + list(cluster_boundaries[:-1])
    cluster_centers = [(start + start + count - 1) / 2 for start, count in zip(cluster_start_indices, counts)]

    # ---- 🔹 Heatmap ----
    sns.heatmap(
        sorted_scores,
        cmap='Greys',
        xticklabels=False,
        yticklabels=False,
        cbar_ax=ax_cbar,  # 👈 use the custom colorbar axis
        cbar_kws={"label": "LRP Contribution"},
        ax=ax
    )

    # ✅ Left-side cluster color blocks
    for i, cluster in enumerate(sorted_clusters):
        ax.add_patch(plt.Rectangle(
            (-1.5, i), 1.5, 1,
            linewidth=0,
            facecolor=to_rgba(CLUSTER_COLORS.get(cluster, '#FFFFFF')),
            clip_on=False
        ))

    # ✅ Cluster counts on left
    for cluster_id, center_y, count in zip(unique_clusters, cluster_centers, counts):
        ax.text(
            -2.0, center_y, f"{count}",
            va='center', ha='right', fontsize=12, fontweight='bold'
        )

    # ✅ Colored x-axis labels
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(pretty_labels, rotation=90, fontsize=12)
    for label, color in zip(ax.get_xticklabels(), feature_colors):
        label.set_color(color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # ---- 🔹 Omics legend ----
    ax_legend.axis("off")
    omics_patches = [
        Patch(color=color, label=omics.upper()) for omics, color in omics_colors.items()
    ]
    ax_legend.legend(
        handles=omics_patches,
        loc="center",
        ncol=len(omics_patches),
        frameon=False
    )


    # ---- 🔹 Filled LRP curve between heatmap and colorbar ----
    lrp_sums = sorted_scores.sum(axis=1)
    lrp_sums = (lrp_sums - lrp_sums.min()) / (lrp_sums.max() - lrp_sums.min())

    y = np.arange(len(lrp_sums))
    ax_curve.fill_betweenx(
        y, 0, lrp_sums,
        color='slategray',
        alpha=0.5,
        linewidth=0
    )

    # ✅ Add "0" and "1" above the curve
    ax_curve.set_xlim(0, 1)
    ax_curve.set_xticks([0, 1])
    ax_curve.set_xticklabels(['0', '1'], fontsize=10)
    ax_curve.tick_params(axis='x', direction='out', pad=1)
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.set_ticks_position('top')
    ##ax_curve.set_xlabel("LRP Sum (Normalized)", fontsize=10, labelpad=4)

    # ✅ Keep y-limits and hide frame
    ax_curve.set_ylim(0, len(lrp_sums))
    ax_curve.spines['right'].set_visible(False)
    ax_curve.spines['left'].set_visible(False)
    ax_curve.spines['bottom'].set_visible(False)
    ax_curve.spines['top'].set_visible(False)
    ax_curve.tick_params(axis='y', length=0)

    
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved spectral clustering heatmap to {output_path}")
