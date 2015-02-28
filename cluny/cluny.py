import cluny_source
import cluny_preprocess
from pylab import *
from scipy.spatial.distance import squareform 
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx

def make_correlation_heatmap(data):
    make_heatmap(cluny_preprocess.correlation_matrix(data))

def make_cross_correlation_heatmap(data):
    make_heatmap(cluny_preprocess.cross_correlation_matrix(data))

def make_correlation_adjacency_heatmap(data):
    make_heatmap(cluny_preprocess.correlation_adjacency_matrix(data))

def make_cross_correlation_adjacency_heatmap(data):
    make_heatmap(cluny_preprocess.cross_correlation_adjacency_matrix(data))

def make_heatmap(data):
    pcolor(data)
    colorbar()
    show()

def make_dendrogram(ccm):
    linkage_matrix = linkage(np.triu(ccm), "single")
    ddata = dendrogram(linkage_matrix,
               show_leaf_counts=False, color_threshold=.7,
               orientation='left',
               count_sort='ascending'
               )
    show()

def make_good_heatmap(D):
    data_dist = 1. - D
    np.fill_diagonal(data_dist, 0.)
    data_dist = squareform(data_dist)


    # Compute and plot first dendrogram.
    fig = plt.figure()
    # x ywidth height
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = linkage(data_dist, method='complete')
    Z1 = dendrogram(Y, orientation='right',  color_threshold=.3)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Z2 = dendrogram(Y, color_threshold=.3)
    ax2.set_xticks([])
    ax2.set_yticks([])

    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    show()

def make_correlation_graph(mat):
    G = cluny_preprocess.correlation_graph(mat)
    nx.draw(G)
    print(len(nx.edges(G)))
    show()

def make_cross_correlation_graph(G, positions):
    nx.draw(G, positions)
    print(len(nx.edges(G)))
    show()

positions = {}


for mat in cluny_source.generate_source():
    ccm = cluny_preprocess.cross_correlation_matrix(mat)
    make_good_heatmap(ccm)
    ccam = cluny_preprocess.cross_correlation_adjacency_matrix(ccm)
    make_heatmap(ccam)
    G = nx.Graph(ccam)
    if not positions:
        positions = nx.spring_layout(G)
    make_cross_correlation_graph(G, positions)
