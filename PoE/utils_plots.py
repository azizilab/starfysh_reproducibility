import seaborn as sns
import matplotlib.pyplot as plt


def plot_gene_dist(data, gene):
    """
    data: data frame
    gene: in columns
    """
    plt.figure(dpi=100)
    sns.histplot(data=data,x=gene)