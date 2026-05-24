import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def plot_comparison_table(results, modality_1, modality_2, out_dir):
    """
    Renders a comparison table of SSIM, PSNR, MSE across models.
    Best value in each metric column is highlighted green.
    Modality pair shown at the top.
    """
    if not results:
        print("No results to plot.")
        return

    models  = [r['model'] for r in results]
    ssims   = [r['ssim']  for r in results]
    psnrs   = [r['psnr']  for r in results]
    mses    = [r['mse']   for r in results]

    # Best = highest SSIM, highest PSNR, lowest MSE
    best_ssim = max(range(len(ssims)), key=lambda i: ssims[i])
    best_psnr = max(range(len(psnrs)), key=lambda i: psnrs[i])
    best_mse  = min(range(len(mses)),  key=lambda i: mses[i])

    col_labels  = ['Model', 'SSIM ↑', 'PSNR ↑ (dB)', 'MSE ↓']
    cell_text   = [
        [m, f"{s:.4f}", f"{p:.2f}", f"{e:.6f}"]
        for m, s, p, e in zip(models, ssims, psnrs, mses)
    ]

    # Build colour grid — white by default, green for best
    HIGHLIGHT = '#90EE90'   # light green
    WHITE     = '#FFFFFF'
    HEADER_BG = '#2C3E50'

    n_rows = len(results)
    colors = [[WHITE] * 4 for _ in range(n_rows)]
    colors[best_ssim][1] = HIGHLIGHT
    colors[best_psnr][2] = HIGHLIGHT
    colors[best_mse][3]  = HIGHLIGHT

    # Alternate row shading for readability
    STRIPE = '#F2F2F2'
    for i in range(n_rows):
        for j in range(4):
            if colors[i][j] == WHITE and i % 2 == 1:
                colors[i][j] = STRIPE

    fig_height = max(3, 1.2 + n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis('off')

    title = f"Synthetic MRI Comparison — {modality_1} → {modality_2}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98,
                 color=HEADER_BG)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=colors,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header row
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color='white', fontweight='bold')

    # Bold the best cells
    for row_idx, col_idx in [(best_ssim + 1, 1),
                              (best_psnr + 1, 2),
                              (best_mse  + 1, 3)]:
        table[row_idx, col_idx].set_text_props(fontweight='bold')

    # Legend
    legend_patch = mpatches.Patch(color=HIGHLIGHT, label='Best value in metric')
    fig.legend(handles=[legend_patch], loc='lower right',
               fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = Path(out_dir) / 'comparison_table.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison table saved to {out_path}")