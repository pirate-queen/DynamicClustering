# Contains functions used for visualizations and analysis.

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