import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
TOOLS_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-1])
sys.path.append(TOOLS_DIR)

COLORS = ['royalblue', 'darkorange', 'slategray', 'orangered', 'darkslateblue',
    'goldenrod', 'chocolate', 'darkorchid', 'olive', 'skyblue', 'lightpink',
    'burlywood', 'mediumseagreen', 'plum', ]

COLORS2 = ['royalblue', 'darkorange', 'mediumseagreen', 'orangered', 'darkslateblue',
    'goldenrod', 'chocolate', 'darkorchid', 'olive', 'skyblue', 'lightpink',
    'burlywood', 'plum', 'slategray']
# colorblind colors
COLORS3 = ['royalblue', 'darkorange', 'olive', 'darkslateblue', 'darkgray', 'lightskyblue']
BAR_COLOR = "#050505"

ALPHA = 0
BASE = 0x40
LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]
MARKERS = [".", "^", "v", "*", "h", "D", "<", ">"]
MARKERS = [".", ",", "^", "v", "*", "h", "<", ">"]

tightargs = {
    "bbox_inches": 'tight',
    "pad_inches": 0.05
}

errorbarargs = {'capsize': 10,
    'ecolor': '#404040', 
    'fmt': 'none',
    'elinewidth': 2,
    'capthick': 2 }

def get_plot_style(i, alpha=ALPHA, base=BASE):
    if i >= len(LINESTYLES)*len(COLORS2):
        raise ValueError(f"Too many linestyles were required: {i}")
    color = COLORS2[i%len(COLORS2)]
    color = np.array([int(matplotlib.colors.cnames[color][x+1:x+3], base=16) for x in [0, 2, 4]])
    color = (color*(1-alpha)+base*alpha).astype(int)
    color = f"#{color[0]:>02x}{color[1]:>02x}{color[2]:>02x}"
    return color, LINESTYLES[i // len(COLORS2)]

def get_scatter_style(i, alpha=ALPHA, base=BASE):
    if i >= len(MARKERS)*len(COLORS2):
        raise ValueError(f"Too many scatter style were required: {i}")
    color = COLORS2[i%len(COLORS2)]
    color = np.array([int(matplotlib.colors.cnames[color][x+1:x+3], base=16) for x in [0, 2, 4]])
    color = (color*(1-alpha)+base*alpha).astype(int)
    color = f"#{color[0]:>02x}{color[1]:>02x}{color[2]:>02x}"
    
    return color, MARKERS[i // len(COLORS2)]

def set_font():
    # cache_path = f"{os.environ['HOME']}/.cache/matplotlib/fontlist-v330.json"
    # if os.path.exists(cache_path):
    #     print("cache removed.")
    #     os.remove(cache_path)
    # plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 16
setfont = set_font

def draw_hist(ax, xs, ys, color, width=None, label=None, **kwargs):
    if len(xs) > 1:
        if width is not None:
            print(f"[WARNING] 'width' argument is overwritten by xs[1] - xs[0]")
        width = xs[1] - xs[0]
    assert len(xs) == len(ys), f"len(xs) ({len(xs)}) != len(ys) ({len(ys)})"
    ax.bar(xs, ys, color=color, alpha=0.2, width=width, **kwargs)
    lxs = [xs[0]-width/2]
    lys = [0]
    for x, y in zip(xs, ys):
        lxs += [x-width/2, x+width/2]
        lys += [y, y]
    lxs.append(xs[-1]+width/2)
    lys.append(0)
    ax.plot(lxs, lys, color=color, label=label)
    return ax

def draw_signif(ax, i_left, i_right, data, bar_xs, bar_ys, 
    signif_margin=None, signif_height=None, maxasterix=None, linewidth=3, color='black', **text_kwargs):

    if '__len__' in dir(i_left):
        i_lefts, i_rights, datas = i_left, i_right, data
    else:
        i_lefts, i_rights, datas = [i_left], [i_right], [data]

    if signif_margin is None:
        signif_margin = max(bar_ys)*0.1
    if signif_height is None:
        signif_height = max(bar_ys)*0.1

    max_bar_height = -np.inf
    for i_signif, (i_left, i_right, data) in enumerate(zip(i_lefts, i_rights, datas)):
        if type(data) is str:
            text = data
        else:
            # * is p < 0.05
            # ** is p < 0.005
            # *** is p < 0.0005
            # etc.
            text = ''
            p = .05
            while data < p:
                text += '*'
                p /= 10.
                if maxasterix and len(text) == maxasterix:
                    break
            if len(text) == 0:
                text = 'n. s.'

        height = max(np.array(bar_ys)[i_left:i_right+1].max(), max_bar_height)
        height += signif_margin
        

        lx, rx = bar_xs[i_left], bar_xs[i_right]

        barx = [lx, lx, rx, rx]
        bary = [height, height+signif_height, height+signif_height, height]
        mid = ((lx+rx)/2, height+signif_height)
    
        ax.plot(barx, bary, c=color, linewidth=linewidth)
        if 'color' not in text_kwargs:
            text_kwargs['color'] = color
        ax.text(*mid, text, ha='center',va='bottom', **text_kwargs)
        max_bar_height = height+signif_height

def savegdrive(fig, figname, folder_path_or_id, ext='png', *args, **kwargs):
    temp_file = f"temp_{figname}.{ext}"
    fig.savefig(temp_file, *args, **kwargs)
    figname_ext = figname.split('.')[-1]
    if figname_ext != ext:
        figname = f"{figname}.{ext}"
    upgdrive(temp_file, figname, mimetype=f"image/{ext}", folder_path_or_id=folder_path_or_id)
    os.remove(temp_file)



