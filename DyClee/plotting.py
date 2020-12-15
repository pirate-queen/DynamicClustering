import numpy as np 
import pandas as pd 
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#%matplotlib inline
import seaborn as sns
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

# Contains functions used for visualizations and analysis.
import numpy as np

# Function to return Xmin and ymin for rectangle parameters
def hyperbox_coordinates(uC, sizes):
	center = uC.get_center() 
	Xmin = (center[0] - sizes[0]/2)
	ymin = (center[1] - sizes[1]/2)
	return (Xmin, ymin)

# Function to unpack and sort snapshot dictionary after clustering process.
# @param snapshots	A snapshots dictionary from a DyClee instance
# @return			Returns an ordered list of timestamps and a corresponding
#					"flattened" snapshot dictionary.
def unpack_snapshots(snapshots):
	snapshots_ordered = {}
	for order, timestamps in snapshots.items(): 
		for timestamp, cluster_lists in timestamps.items(): 
			snapshots_ordered[timestamp] = cluster_lists
	return sorted(snapshots_ordered), snapshots_ordered

# Function to plot snapshots after unpacking 
# @param timestamp_order	A list of sorted snapshot times
# @param snapshots_ordered	A stripped snapshot dictionary that does not contain the order key 
# @param display_class		An optional parameter that displays the clustered class on each data point
def plot_snapshots(timestamp_order, snapshots_ordered, display_class=False): 

	fig, axes = plt.subplots(len(timestamp_order), 2, sharex='col', sharey='row', figsize=(20,600))
	cols = ["FinalClusters", "MicroClusters"]
	for ax, col in zip(axes[0], cols): ax.set_title(col)

	for i, t in enumerate(timestamp_order): 
		final_list = snapshots_ordered[t]['final']
		micro_list = snapshots_ordered[t]['all']

		# plot final cluster
		final_df = pd.DataFrame([np.append(uC.center, [uC.label], 0) for uC in final_list], columns=['x', 'y', 'class'])
		f_plot = sns.scatterplot(ax=axes[i][0], x='x',y='y',hue='class',data=final_df)
		f_plot.legend(loc='lower left', bbox_to_anchor=(1.05,0), ncol=1)

		if (display_class): 
			for line in range(0,final_df.shape[0]): 
				f_plot.text(final_df['x'][line], final_df['y'][line], final_df['class'][line], horizontalalignment='left', size='medium', color='black', weight='normal')
		axes[i][0].set_ylabel(t, rotation=0, size='xx-large', weight='bold')
		
		# plot micro clusters 
		micro_df = pd.DataFrame([np.append(uC.get_center(), [uC.Classk], 0) for uC in micro_list], columns=['x', 'y', 'class'])
		m_plot = sns.scatterplot(ax=axes[i][1], x='x',y='y',hue='class',data=micro_df)
		m_plot.legend(loc='lower left', bbox_to_anchor=(1.05,0), ncol=1)
		if (display_class): 
			for line in range(0,micro_df.shape[0]): 
				if micro_df['class'][line] != 'Unclassed': 
					m_plot.text(micro_df['x'][line], micro_df['y'][line], micro_df['class'][line], horizontalalignment='left', size='medium', color='black', weight='normal')

	fig.tight_layout

# Function to get the context matrix of a dataset 
# @param X: Data 
# @return:	returns a context matrix, with the mins in row 1 and max in row 2
def get_context_matrix(X): 
	return np.vstack([X.min(axis=0), X.max(axis=0)])

# Function to plot the hyperboxes of given microclusters 
# @param all_uC		A list of all microclusters. Combination of the A-list and O-list 
# @ hyperbox_size	dimensions of the hyperbox size. Should be in the format (x, y) for 2d data
def plot_hyperboxes(all_uC, hyperbox_size): 

	cluster_df = pd.DataFrame([uC.get_center() for uC in all_uC], columns=['x', 'y'])
	sns.scatterplot(x='x',y='y',data=cluster_df)
	plt.gca().set_aspect('equal')

	for uC in all_uC: 
		xy = hyperbox_coordinates(uC, hyperbox_size)
		plt.gca().add_patch(Rectangle(xy,hyperbox_size[0],hyperbox_size[1],linewidth=0.5,edgecolor='r',facecolor='none', clip_on=False))

	return plt

# Function to plot density of microclusters 
# @param all_uC		A list of all microclusters
def plot_density(all_uC): 

	cluster_df = pd.DataFrame([uC.get_center() for uC in all_uC], columns=['x', 'y'])
	cluster_class = np.array([uC.density_type for uC in all_uC])

	# plot microclusters
	sns.scatterplot(x='x',y='y',hue=cluster_class,data=cluster_df).legend(loc='lower left', bbox_to_anchor=(1.05,0), ncol=1)
	plt.gca().set_aspect('equal')

# Function to test the dyclee algorithm on test datasets
# @param datasets:	A list of datasets. Should be in the format [ [[data],[labels]], [[..],[..]], .. ] 
# @param dyclee:	dyclee object 
def test_dyclee_datasets(datasets, dyclee): 
	fig, ax = plt.subplots(len(datasets), 2, figsize=(10,10))

	# assign column names 
	cols = ["Original", "Dyclee"]
	for axes, col in zip(ax[0], cols): axes.set_title(col)

	for i, data in enumerate(datasets):
		
		# get and standardize data 
		X = data[0]
		y = data[1]
		Xx = StandardScaler().fit_transform(X)

		original_df = pd.DataFrame(Xx, columns=['x', 'y'])
		sns.scatterplot(ax=ax[i][0], x='x',y='y',hue=y,data=original_df, palette='bright').legend(loc='lower left', bbox_to_anchor=(1.05,0), ncol=1)

		y_pred = dyclee.run_dataset(data=X, targetcol=y) 
		all_uC = dyclee.A_list + dyclee.O_list

		cluster_df = pd.DataFrame([uC.get_center() for uC in all_uC], columns=['x', 'y'])
		output_y = np.array([uC.Classk for uC in all_uC]) 
		sns.scatterplot(ax=ax[i][1], x='x',y='y',hue=output_y,data=cluster_df,palette='bright').legend(loc='lower left', bbox_to_anchor=(1.05,0), ncol=1)

	fig.tight_layout()

# Function for stripping the interal DyClee generated label prefix from labels.
# Useful for shortening labels in an unsupervised learning scenario.
def strip_labels(labels):
	return np.array([label.split("_")[1] if label != "Unclassed" else label \
		for label in labels])

