import cluny_source
import cluny_preprocess
from pylab import *
from scipy.spatial.distance import squareform 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import networkx as nx
import pandas as pd

def make_correlation_heatmap(data):
    make_heatmap(cluny_preprocess.correlation_matrix(data))

def make_cross_correlation_heatmap(data):
    make_heatmap(cluny_preprocess.cross_correlation_matrix(data))

def make_correlation_adjacency_heatmap(data):
    make_heatmap(cluny_preprocess.correlation_adjacency_matrix(data))

def make_cross_correlation_adjacency_heatmap(data):
    make_heatmap(cluny_preprocess.cross_correlation_adjacency_matrix(data))

def make_heatmap(data, name):
    pcolor(data)
    colorbar()
    savefig(name+"_am.png", format="png")
    show()

def make_dendrogram(ccm):
    linkage_matrix = linkage(np.triu(ccm), "single")
    ddata = dendrogram(linkage_matrix,
               show_leaf_counts=False, color_threshold=.7,
               orientation='left',
               count_sort='ascending'
               )
    show()

def make_good_heatmap(D, name):
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
    savefig(name + "_hm.png", format="png")
    show()

def make_correlation_graph(mat):
    G = cluny_preprocess.correlation_graph(mat)
    nx.draw(G)
    print(len(nx.edges(G)))
    show()

def plot_mat(data, name):
    # Three subplots sharing both x/y axes
    f, axarray = plt.subplots(8, sharex=True, sharey=True)
    for i, row in enumerate(data):
        axarray[i].plot(range(len(row)), row, 'g')
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
    axarray[0].set_title('10 second EEG Reading for Patient')
    axarray[int(len(axarray) / 2)].set_ylabel('Magnitude')
    axarray[-1].set_xlabel('Time')
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    plt.rc('font', **font)
    savefig(name + "_series.png", format="png")
    plt.show()


class NetworkPlot:
    def __init__(self):
        self.positions = {}
        for i in range(76):
            if i < 64:
                row = np.floor(i / 8)
                col = i % 8
            else:
                x = i - 64
                row = np.floor(x / 6) + 8
                col = x % 6
            self.positions[i] = (row, col)

    def make_cross_correlation_graph(self, G):
        nx.draw(G, self.positions)
        print(len(nx.edges(G)))
        show()

    def make_vertex_degree_centrality(self, G, name, color="b"):
        deg = nx.degree_centrality(G)
        name = name + "_deg"
        self.make_custom_plot(G, deg, name, color=color)

    def make_closeness_centrality(self, G, name, color="b"):
        close = nx.closeness_centrality(G)
        name = name + "_close"
        self.make_custom_plot(G, close, name, color=color)

    def make_betweenness_centrality(self, G, name, color="b"):
        bet = nx.betweenness_centrality(G)
        name = name + "_bet"
        self.make_custom_plot(G, bet, name, color=color)

    def make_clustering_coefficient(self, G, name, color="b"):
        cc = nx.clustering(G)
        name = name + "_cc"
        self.make_custom_plot(G, cc, name, color=color)

    def make_custom_plot(self, G, nodesize, name, color="b"):
        nodesize = np.array([np.sqrt(nodesize[v])+.001 for v in G])*1000
        nx.draw_networkx_nodes(G, self.positions, node_color=color, node_size=nodesize)
        savefig(name + "_graph.png", format="png")
        show()

    def make_cluster_plot(self, G, name, colors):
        nodesize = nx.degree_centrality(G)
        nodesize = np.array([np.sqrt(nodesize[v])+.001 for v in G])*1000
        nx.draw_networkx_nodes(G, self.positions, node_color=colors, node_size=nodesize)
        nx.draw_networkx_edges(G, self.positions)
        savefig(name + "_clusters.png", format="png")
        show()

class DifferenceNetworkPlot(NetworkPlot):
    def __init__(self):
        super().__init__()
        self.degs = []
        self.closes = []
        self.bets = []
        self.ccs = []



    def make_vertex_degree_centrality(self, G1, G2, name):
        deg1 = nx.degree_centrality(G1)
        deg2 = nx.degree_centrality(G2)
        name = name + "_deg"
        self.make_difference(G1, name, deg1, deg2)

    def make_closeness_centrality(self, G1, G2, name):
        close1 = nx.closeness_centrality(G1)
        close2 = nx.closeness_centrality(G2)
        name = name + "_close"
        self.make_difference(G1, name, close1, close2)

    def make_betweenness_centrality(self, G1, G2, name):
        bet1 = nx.betweenness_centrality(G1)
        bet2 = nx.betweenness_centrality(G2)
        name = name + "_bet"
        self.make_difference(G1, name, bet1, bet2)

    def make_clustering_coefficient(self, G1, G2, name):
        cc1 = nx.clustering(G1)
        cc2 = nx.clustering(G2)
        name = name + "_cc"
        self.make_difference(G1, name, cc1, cc2)

    def make_difference(self, G, name, nodesize1, nodesize2):
        nodesize1 = np.array([np.sqrt(nodesize1[v])+.001 for v in G])*1000
        nodesize2 = np.array([np.sqrt(nodesize2[v])+.001 for v in G])*1000
        diff = nodesize1 - nodesize2
        name = name + "_diff"
        pos = [i for i, d  in enumerate(diff) if d >= 0]
        neg = [i for i, d  in enumerate(diff) if d < 0]
        self.make_custom_plot(G, diff, pos, neg, name)

    def make_custom_plot(self, G, diff, pos, neg, name):
        nodesize = np.array([np.absolute(d) + 1 for d in diff])
        nx.draw_networkx_nodes(G, self.positions, node_color="g", nodelist=pos, node_size=nodesize)
        nx.draw_networkx_nodes(G, self.positions, node_color="r", nodelist=neg, node_size=nodesize)
        savefig(name + "_graph.png", format="png")
        show()



#nx_plot = NetworkPlot()
#for mat, name in cluny_source.generate_source():
#    name = name.split('.')[0]
#    plot_mat(mat[:8], name)
    #ccm = cluny_preprocess.cross_correlation_matrix(mat)
    #make_good_heatmap(ccm, name)
    #ccam = cluny_preprocess.cross_correlation_adjacency_matrix(ccm)
    #make_heatmap(ccam, name)
    #G = nx.Graph(ccam)        
    #nx_plot.make_degree_centrality(G, name)
def make_ps(nx_plot, G, name, color='g'):
    nx_plot.make_vertex_degree_centrality(G, name, color=color)
    nx_plot.make_closeness_centrality(G, name, color=color)
    nx_plot.make_betweenness_centrality(G, name, color=color)
    nx_plot.make_clustering_coefficient(nx.Graph(G), name, color=color)

def make_diff_ps(nx_diff_plot, G1, G2, name):
    nx_diff_plot.make_vertex_degree_centrality(G1, G2, name)
    nx_diff_plot.make_closeness_centrality(G1, G2, name)
    nx_diff_plot.make_betweenness_centrality(G1, G2, name)
    nx_diff_plot.make_clustering_coefficient(nx.Graph(G1), nx.Graph(G2), name)

def degree_plot(G1, G2, name):
    degree_sequence1 = sorted(nx.degree(G1).values(),reverse=True) # degree sequence
    degree_sequence2 = sorted(nx.degree(G2).values(),reverse=True) # degree sequence
    #print "Degree sequence", degree_sequence
    plt.plot(degree_sequence1, 'g', marker='o')
    plt.plot(degree_sequence2, 'r', marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
    plt.savefig("degree_plot" + name + ".png")
    plt.show()

def degree_distribution(G1, G2, name):
    degrees1 = G1.degree() # dictionary node:degree
    values1 = sorted(set(degrees1.values()))
    hist1 = [list(degrees1.values()).count(x) for x in values1]
    degrees2 = G2.degree() # dictionary node:degree
    values2 = sorted(set(degrees2.values()))
    hist2 = [list(degrees2.values()).count(x) for x in values2]
    fig, ax = plt.subplots()
    plt.plot(values1,hist1,'go-') # in-degree
    plt.plot(values2,hist2,'rv-') # out-degree
    plt.legend(['ictal-degree','preictal-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    ax.set_ylim(0, 12)
    plt.savefig("degree_distribution_plot" + name + ".png")
    plt.close()

def return_color_list(ccm):
    dist_ccm = 1. - ccm
    np.fill_diagonal(dist_ccm, 0.)
    dist_ccm = squareform(dist_ccm)
    Y = linkage(dist_ccm, method='complete')
    return fcluster(Y, .3)

def dcg(relevances, rank=20):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg(relevances, rank=20):
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
    return dcg(relevances, rank) / best_dcg

#nx_plot = NetworkPlot()
#nx_diff_plot = DifferenceNetworkPlot()
#GM1 = nx.MultiGraph()
#GM2 = nx.MultiGraph()
#GM1.add_nodes_from([i for i in range(76)])
#GM2.add_nodes_from([i for i in range(76)])
#for i, (mat, name) in enumerate(cluny_source.generate_source()):
#    name = name.split('.')[0]
#    ccm = cluny_preprocess.cross_correlation_matrix(mat)
#    Z1 = return_color_list(ccm)
#    ccam = cluny_preprocess.cross_correlation_adjacency_matrix(ccm)
#    if i%2 == 0:
#        G1 = nx.Graph(ccam)
#        #GM1.add_edges_from(G1.edges())
#        #make_ps(nx_plot, G1, name, color='g')
#    else:
#        G2 = nx.Graph(ccam)
        #degree_distribution(G1, G2, name)
        #GM2.add_edges_from(G2.edges())
        #make_ps(nx_plot, G2, name, color='r')
        #make_diff_ps(nx_diff_plot, G1, G2, name)

#make_ps(nx_plot, GM1, 'all_g1', color='g')
#make_ps(nx_plot, GM2, 'all_g2', color='r')
#make_diff_ps(nx_diff_plot, GM1, GM2, 'all_diff')

nx_plot = NetworkPlot()
nx_diff_plot = DifferenceNetworkPlot()
GM1 = nx.MultiGraph()
GM2 = nx.MultiGraph()
GM1.add_nodes_from([i for i in range(76)])
GM2.add_nodes_from([i for i in range(76)])
df = pd.DataFrame()
col_max = pd.Series()
for i, (mat, name) in enumerate(cluny_source.generate_source()):
    name = name.split('.')[0]
    ccm = cluny_preprocess.cross_correlation_matrix(mat)
    colors = return_color_list(ccm)
    col_max[name] = colors.max()
    ccam = cluny_preprocess.cross_correlation_adjacency_matrix(ccm)
    G = nx.Graph(ccam)
    nx_plot.make_cluster_plot(G, name, colors)
    df[name + "-deg"] = pd.Series(nx.degree_centrality(G))
    df[name + "-bet"] = pd.Series(nx.betweenness_centrality(G))
    df[name + "-close"] = pd.Series(nx.closeness_centrality(G))
    df[name + "-cc"] = pd.Series(nx.clustering(G))
