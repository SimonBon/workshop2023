import numpy as np
import matplotlib.pyplot as plt

def rand_col_seg(seg) -> np.ndarray:
    
    vals = np.unique(seg)
    colors = np.random.uniform(0.1, 1, (vals.max()+1, 3))
    colors[0] = [0, 0, 0]

    return colors[seg]


def gridPlot(ims, labels=None, targets=None, sz=(10,10), vmin=0, vmax=1, save_path=None, plot=True, title=None, fontsize=12, figscale=2):
    
    fig, axs = plt.subplots(sz[0], sz[1], figsize=(figscale*sz[1], figscale*sz[0]))
    
    for n, (ax, im) in enumerate(zip(axs.ravel(), ims[:sz[0]*sz[1]])):
        
        ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if isinstance(labels, (list, np.ndarray)) and isinstance(targets, (list, np.ndarray)):
            ax.set_title([labels[n], targets[n]], fontsize=fontsize)
        elif isinstance(labels, (list, np.ndarray)):
            ax.set_title(labels[n], fontsize=fontsize)
        else:
            ax.set_title(n)
      
    if isinstance(title, str):       
        fig.suptitle(title, fontsize=20)
    
    plt.tight_layout()
        
    if isinstance(save_path, str):
        plt.savefig(save_path)
        plt.close(fig)
    if plot: 
        plt.show()