"""Plotting tools for asteroid hammer"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patheffects as path_effects
import lightkurve as lk

def two_panel_movie(data_a, data_b, out='out.mp4', scale='linear',
                    title_a='', title_b='', **kwargs):
    '''Create an mp4 movie of a 3D array
    '''
    if scale == 'log':
        data_A = np.log10(np.copy(data_a), stack)
        data_B = np.log10(np.copy(data_b), stack)
    else:
        data_A = np.copy(data_a)
        data_B = np.copy(data_b)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax in axs:
        ax.set_facecolor('#ecf0f1')
    im1 = axs[0].imshow(data_A[0], origin='bottom', **kwargs)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title(title_a, fontsize=10)
    im2 = axs[1].imshow(data_B[0], origin='bottom', ** kwargs)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title(title_b, fontsize=10)
    def animate(i):
        im1.set_array(data_A[i])
        im2.set_array(data_B[i])
    anim = animation.FuncAnimation(fig, animate, frames=len(data_A), interval=30)
    anim.save(out, dpi=150)
    del data_A, data_B


def movie(data, asteroids={}, out='out.mp4', scale='linear',
                    title='', **kwargs):
    # This makes things faster
    if isinstance(data, lk.targetpixelfile.TargetPixelFile):
        shape = data.shape
        data = np.copy(data.flux)
    elif isinstance(data, np.ndarray):
        shape = data.shape
    else:
        raise ValueError('can not parse input')

    if len(asteroids) != 0:
        for label in ['column', 'row', 'names']:
            if label not in asteroids:
                raise ValueError('can not parse `asteroids`')
            if len(asteroids[label]) != len(data):
                if label is 'names':
                    continue
                raise ValueError(f'asteroid {label} input is not same length as data')
        c = asteroids['column']
        r = asteroids['row']
        names = asteroids['names']

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.set_facecolor('#ecf0f1')
    im = ax.imshow(data[0], origin='bottom', **kwargs)
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    if len(asteroids) != 0:
        scat = ax.scatter(c[0], r[0], facecolor='None', s=100, edgecolor='r')
        scat.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                               path_effects.Normal()])
        texts = [ax.text(-10, -10, '', color='red', size=6, ha='left', va='center') for idx in range(c.shape[1])]
        [text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                               path_effects.Normal()]) for text in texts]

    ax.set(xlim=xlims, ylim=ylims)
    ax.set_xticks([])
    ax.set_yticks([])
    def animate(i):
        im.set_array(data[i])
        if len(asteroids) != 0 :
            scat.set_offsets(np.vstack([c[i], r[i]]).T)
            for jdx in range(c.shape[1]):
                coords = c[i][jdx], r[i][jdx]
                if ((coords[0] > 10) & (coords[0] < (shape[1] - 10)) &
                    (coords[1] > 10) & (coords[1] < (shape[2] - 10))):
                    texts[jdx].set_x(coords[0])
                    texts[jdx].set_y(coords[1])
                    texts[jdx].set_text(names[jdx])
                else:
                    texts[jdx].set_x(-10)
                    texts[jdx].set_y(-10)
                    texts[jdx].set_text('')

        if len(asteroids) != 0:
            return im, scat, texts,
        return im
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=30)
    anim.save(out, dpi=150)
