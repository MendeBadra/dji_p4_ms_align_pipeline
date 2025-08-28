import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.cluster import KMeans
from pathlib import Path
import imageio.v3 as iio
from esda.moran import Moran
from libpysal.weights import lat2W

def compute_morans_i(image: np.ndarray, use_rook: bool = True, nan_to_zero: bool = True) -> tuple[float, float]:
    """
    Compute Moran's I statistic for a 2D image array.

    Parameters:
    -----------
    image : np.ndarray
        2D array of image values (e.g., MSAVI values).
    use_rook : bool
        Whether to use rook (4-neighbor) adjacency. If False, queen (8-neighbor) is used.
    nan_to_zero : bool
        Whether to replace NaNs with zeros.

    Returns:
    --------
    morans_i : float
        The Moran's I value.
    p_value : float
        The pseudo p-value from randomization test.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    # Replace NaNs if required
    if nan_to_zero:
        image = np.nan_to_num(image)

    rows, cols = image.shape
    x = image.flatten()

    # Build spatial weights
    w = lat2W(rows, cols, rook=use_rook)

    # Compute Moran's I
    mi = Moran(x, w)

    return mi.I, mi.p_sim

def mask_msavi(msavi_data, initial_threshold):
    relevant_pixels_mask = msavi_data >= initial_threshold # not ground
    msavi_nonnan = np.nan_to_num(msavi_data, nan=-1.0)
    # msavi_masked = msavi_nonnan if ~relevant_pixels_mask is None else np.where(relevant_pixels_mask, msavi_nonnan, -1.0)
    msavi_masked = np.where(relevant_pixels_mask, msavi_nonnan, -1.0)
    return msavi_masked

def segment_msavi_by_kmeans(msavi_data, initial_threshold, k=3):
    """
    Segments MSAVI data to find the cluster with the highest mean value.

    Args:
        msavi_data (np.ndarray): Input MSAVI image data.
        initial_threshold (float): A threshold to create an initial mask.
                                   Only pixels above this threshold are considered for KMeans.
        k (int): Number of clusters for KMeans.

    Returns:
        tuple: (weed_mask, weed_cluster_value, kmeans_model)
               - weed_mask (np.ndarray): Boolean mask of the weed cluster.
               - weed_cluster_value (float): The mean intensity of the weed cluster.
                                             Returns None if no weeds found.
               # DEPRECATED: - kmeans_model (KMeans): Fitted KMeans model. Returns None if clustering failed.
               - Now returns the list of cluster masks sorted by centroid values.
    """
    h, w = msavi_data.shape
    
    # Apply initial threshold to focus KMeans on relevant vegetation
    # Pixels below threshold are considered non-vegetation or less relevant
    print(initial_threshold)
    # relevant_pixels_mask = msavi_data >= initial_threshold # not ground
    # msavi_nonnan = np.nan_to_num(msavi_data, nan=-1.0)
    # msavi_masked = msavi_nonnan if ~relevant_pixels_mask is None else np.where(relevant_pixels_mask, msavi_nonnan, -1.0)# msavi_nonnan * relevant_pixels_mask
    msavi_masked = mask_msavi(msavi_data, initial_threshold)

    pixels_for_kmeans = msavi_masked.reshape(-1, 1)

    if pixels_for_kmeans.shape[0] < k: # Not enough data points for k clusters
        print(f"Warning: Not enough data points ({pixels_for_kmeans.shape[0]}) above threshold for {k} clusters. No weed cluster identified.")
        return np.zeros((h, w), dtype=bool), None, None

    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(pixels_for_kmeans)
    except Exception as e:
        print(f"Error during KMeans fitting: {e}")
        return np.zeros((h, w), dtype=bool), None, None

    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_centers = sorted(cluster_centers)  # Optional: for printing or debugging

    # Print all cluster center values for clarity
    print("Cluster centers (sorted by intensity):", sorted_centers)
    
    # Identify the weed cluster (highest mean intensity)
    weed_cluster_value = np.max(cluster_centers)
    
    # Create the full labeled image for all pixels that were part of KMeans
    # Start with an array of a value that won't match any cluster label (e.g., -1)
    all_labels_flat = np.full(msavi_data.size, -1, dtype=int)
    # relevant_indices = np.where(relevant_pixels_mask.flatten())[0]
    all_labels_flat = kmeans.labels_.flatten()
    
    # Weed mask is where the pixel was part of relevant_pixels_mask AND belongs to the weed cluster
    weed_cluster_label = np.argmax(cluster_centers) # Label corresponding to the max center
    weed_mask_flat = (all_labels_flat == weed_cluster_label)
    weed_mask = weed_mask_flat.reshape(h, w)

    labels_img = all_labels_flat.reshape(h, w)
    # Create a mask for each cluster label
    cluster_masks = []
    for label in range(len(cluster_centers)):
        mask = (labels_img == label)
        cluster_masks.append(mask)

    # Optional: print area (number of pixels) in each cluster
    total_area = labels_img.size
    for i, mask in enumerate(cluster_masks):
        area = np.sum(mask)
        print(f"Cluster {i} (center={cluster_centers[i]:.3f}) pixel coverage: {(area / total_area * 100):.2f} %")
    
    return weed_mask, weed_cluster_value, cluster_masks

def plot_msavi_highlighting_weed_cluster(msavi_data, weed_mask, weed_percentage,
                                         output_path: Path, output_prefix: str,
                                         cmap: str = 'viridis', msavi_vmin: float = 1.0, msavi_vmax: float = 1.0):
    """
    Plots the MSAVI image, highlighting only the 'weed' cluster.
    MSAVI values are shown for the weed cluster; other areas are transparent/masked.

    Args:
        msavi_data (np.ndarray): Original MSAVI data.
        weed_mask (np.ndarray): Boolean mask indicating the weed cluster.
        weed_percentage (float): Percentage of the image covered by the weed cluster.
        output_path (Path): Directory to save the plot.
        output_prefix (str): Prefix for the output filename.
        cmap (str): Colormap for displaying MSAVI values.
        msavi_vmin (float): Min value for MSAVI normalization.
        msavi_vmax (float): Max value for MSAVI normalization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a copy of MSAVI data to modify for plotting
    msavi_display = msavi_data.copy()
    
    # Mask out non-weed areas by setting them to NaN (which imshow can render as transparent or background)
    msavi_display[~weed_mask] = np.nan

    norm = Normalize(vmin=msavi_vmin, vmax=msavi_vmax)
    img_plot = ax.imshow(msavi_display, cmap=cmap, norm=norm)

    cbar = fig.colorbar(img_plot, ax=ax, shrink=0.75, label='MSAVI Value (Weed Cluster)')
    
    ax.set_title(f"MSAVI - Weed Cluster Highlight ({output_prefix})\nWeed Coverage: {weed_percentage:.2f}% of total area", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    save_filename = output_path / f"{output_prefix}_msavi_weed_cluster.png"
    try:
        fig.savefig(save_filename, bbox_inches='tight')
        print(f"Plot saved to: {save_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Display the plot


def plot_kmeans_cluster(msavi_data, kmeans_labels, cluster_label, output_path: Path, output_prefix: str,
                        cmap: str = 'viridis', msavi_vmin: float = 1.0, msavi_vmax: float = 1.0):
    """
    Plots the MSAVI image, highlighting pixels belonging to a specific KMeans cluster.

    Args:
        msavi_data (np.ndarray): Original MSAVI 2D array.
        kmeans_labels (np.ndarray): 2D array of KMeans labels with same shape as msavi_data.
        cluster_label (int): Cluster ID to highlight.
        output_path (Path): Directory to save the plot.
        output_prefix (str): Prefix for the output filename.
        cmap (str): Colormap to use.
        msavi_vmin (float): Min value for MSAVI normalization.
        msavi_vmax (float): Max value for MSAVI normalization.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a mask for the selected cluster
    cluster_mask = (kmeans_labels == cluster_label)

    # Create display copy and mask non-cluster values
    msavi_display = msavi_data.copy()
    msavi_display[~cluster_mask] = np.nan

    norm = Normalize(vmin=msavi_vmin, vmax=msavi_vmax)
    img_plot = ax.imshow(msavi_display, cmap=cmap, norm=norm)

    cluster_percentage = 100.0 * np.sum(cluster_mask) / cluster_mask.size

    cbar = fig.colorbar(img_plot, ax=ax, shrink=0.75, label='MSAVI Value (Cluster)')
    
    ax.set_title(f"MSAVI - Cluster {cluster_label} Highlight\nCoverage: {cluster_percentage:.2f}%", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    output_path.mkdir(parents=True, exist_ok=True)
    save_filename = output_path / f"{output_prefix}_cluster_{cluster_label}.png"

    try:
        fig.savefig(save_filename, bbox_inches='tight')
        print(f"Cluster {cluster_label} plot saved to: {save_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show()


def main_process_msavi_for_weeds(msavi_source, threshold, output_dir, file_prefix, k_clusters=3):
    """
    Main function to load MSAVI, process for weeds, and plot.

    Args:
        msavi_source (str | np.ndarray): Path to MSAVI .TIF file or a NumPy array.
        threshold (float): Initial MSAVI threshold for vegetation.
        output_dir (str): Directory to save the output plot.
        file_prefix (str): Prefix for the saved plot filename.
        k_clusters (int): Number of clusters for KMeans.

    Returns:
        the image statistics as dict.
    """
    
    if isinstance(msavi_source, str) or isinstance(msavi_source, Path):
        try:
            msavi_image = iio.imread(msavi_source)
            if msavi_image.ndim == 3:
                msavi_image = msavi_image[:, :, 0] # Assuming MSAVI is first band if multi-band
            elif msavi_image.ndim != 2:
                raise ValueError("MSAVI image must be 2D or 3D with first band as MSAVI.")
            print(f"Loaded MSAVI from: {msavi_source}")
        except Exception as e:
            print(f"Error loading MSAVI image from {msavi_source}: {e}")
            return
    elif isinstance(msavi_source, np.ndarray):
        if msavi_source.ndim == 3 and msavi_source.shape[2] > 0 :
             msavi_image = msavi_source[:, :, 0]
        elif msavi_source.ndim == 2:
             msavi_image = msavi_source
        else:
            print("Error: Input NumPy array for MSAVI must be 2D, or 3D with first band as MSAVI.")
            return
        print("Using provided NumPy array for MSAVI.")
    else:
        print("Error: msavi_source must be a file path (str/Path) or a NumPy array.")
        return

    if msavi_image.size == 0:
        print("Error: MSAVI image data is empty.")
        return

    # Perform segmentation
    weed_cluster_binary_mask, weed_val, kmeans = segment_msavi_by_kmeans(
        msavi_image,
        initial_threshold=threshold,
        k=k_clusters
    )

    msavi_weed_only = np.where(weed_cluster_binary_mask, msavi_image, -1)
    morans_i, p_val = compute_morans_i(msavi_weed_only)
    print(f"Moran's I: {morans_i:.4f}, p-value: {p_val:.4f}")
    output_path_obj = Path(output_dir)
    msavi_masked = mask_msavi(msavi_image, threshold)
    # TEST:
    # morans_i, p_val = compute_morans_i(msavi_masked)
    # print(f"Moran's I: {morans_i:.4f}, p-value: {p_val:.4f}")
    plt.figure(figsize=(10,8))
    plt.imshow(msavi_masked, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')
    plt.title(f'MSAVI masked {file_prefix}')
    plt.tight_layout()
    plt.savefig(output_path_obj / f"{file_prefix}_msavi_masked", bbox_inches="tight")

    if weed_val is None: # No weed cluster identified
        print(f"No weed cluster could be identified for {file_prefix} with threshold {threshold}.")
        # Optionally, plot the original MSAVI or a message
        fig, ax = plt.subplots(figsize=(10,8))
        ax.imshow(msavi_image, cmap='gray') # Show original if no weeds
        ax.set_title(f"MSAVI - No Weed Cluster Identified ({file_prefix})\nThreshold: {threshold}", fontsize=14)
        ax.axis('off')
        
        output_path_obj.mkdir(parents=True, exist_ok=True)
        save_filename = output_path_obj / f"{file_prefix}_msavi_no_weeds.png"
        try:
            fig.savefig(save_filename, bbox_inches='tight')
            print(f"Plot (no weeds) saved to: {save_filename}")
        except Exception as e:
            print(f"Error saving 'no weeds' plot: {e}")
        return

    # Calculate percentage of weed mask relative to total image area
    weed_pixel_count = np.sum(weed_cluster_binary_mask)
    total_pixels = msavi_image.size
    weed_percentage_total_area = (weed_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0

    print(f"Identified weed cluster value: {weed_val:.4f}")
    print(f"Weed cluster coverage: {weed_percentage_total_area:.2f}% of total image area.")

    # Plotting
    output_path_obj = Path(output_dir)
    plot_msavi_highlighting_weed_cluster(
        msavi_data=msavi_image,
        weed_mask=weed_cluster_binary_mask,
        weed_percentage=weed_percentage_total_area,
        output_path=output_path_obj,
        output_prefix=file_prefix,
        # MSAVI typically ranges -1 to 1, but can be scaled (e.g. 0-200 for some sensors, or 0-1 if normalized).
        # Adjust vmin/vmax based on expected MSAVI range from your source TIF files.
        # If your MSAVI is, for example, scaled 0-1:
        msavi_vmin=-1.0, msavi_vmax=1.0
        # msavi_vmin=np.nanmin(msavi_image) if np.any(~np.isnan(msavi_image)) else 0, # Or a fixed typical min like 0.0
        # msavi_vmax=np.nanmax(msavi_image) if np.any(~np.isnan(msavi_image)) else 1  # Or a fixed typical max like 1.0
    )
    # TODO: enable plot for every cluster
    #n_clusters = np.unique(cluster_labels).size
    # for i in range(n_clusters):
    #     plot_kmeans_cluster(msavi_data, cluster_labels, i, Path("outputs"), f"cluster_{i}")

    # TODO: Make this more pretty with function wrapper.
    # Saving the masks to disk
    # Save the weed mask and all cluster masks to disk
    # masks_dir = output_path_obj / "masks"
    # masks_dir.mkdir(parents=True, exist_ok=True)

    # # Save weed mask
    # weed_mask_path = masks_dir / f"{file_prefix}_weed_mask.npy"
    # np.save(weed_mask_path, weed_cluster_binary_mask)
    # print(f"Weed mask saved to: {weed_mask_path}")

    # # Save all cluster masks
    # if kmeans is not None:
    #     for idx, mask in enumerate(kmeans):
    #         mask_path = masks_dir / f"{file_prefix}_cluster{idx}_mask.npy"
    #         np.save(mask_path, mask)
    #         print(f"Cluster {idx} mask saved to: {mask_path}")
    # Collect statistics
    stats = {
        "file_prefix": file_prefix,
        "threshold": threshold,
        "morans_i": morans_i,
        "morans_p_value": p_val,
        "weed_cluster_value": weed_val,
        "weed_coverage_percent": weed_percentage_total_area,
        "k_clusters": k_clusters,
    }

    # Add cluster centers and pixel coverage
    if kmeans is not None:
        for idx, mask in enumerate(kmeans):
            cluster_center = float(np.mean(msavi_image[mask])) if np.any(mask) else float('nan')
            cluster_coverage = float(np.sum(mask)) / msavi_image.size * 100
            stats[f"cluster_{idx}_center"] = cluster_center
            stats[f"cluster_{idx}_coverage_percent"] = cluster_coverage

    plt.close('all')

    return stats



if __name__ == "__main__":
    print("--- Running Example for New Script ---")
    
    # Example Usage:
    # 1. Create a dummy MSAVI .tif file (or use one of your existing files)
    dummy_msavi_data = np.random.rand(100, 100) * 0.5  # MSAVI values mostly between 0 and 0.5
    # Introduce some higher values for potential "weeds"
    dummy_msavi_data[20:40, 20:40] = np.random.rand(20, 20) * 0.3 + 0.6  # Weeds between 0.6 and 0.9
    dummy_msavi_data[70:80, 70:80] = np.random.rand(10, 10) * 0.2 + 0.55 # More weeds
    
    dummy_tif_path = Path("dummy_msavi_example.tif")
    iio.imwrite(dummy_tif_path, dummy_msavi_data.astype(np.float32))
    
    print(f"Dummy MSAVI TIF saved to: {dummy_tif_path}")

    # --- Example 1: Using the dummy TIF file ---
    main_process_msavi_for_weeds(
        msavi_source=str(dummy_tif_path), # Path to your MSAVI .TIF file
        threshold=0.2,                    # MSAVI value above which pixels are considered for clustering
        output_dir="output_plots",        # Directory to save plots
        file_prefix="dummy_sample1",      # Prefix for the output file
        k_clusters=3                      # Number of K-Means clusters
    )

    # --- Example 2: Using a NumPy array directly ---
    # Generate another MSAVI array
    msavi_array_input = np.zeros((120,120))
    msavi_array_input[30:60, 40:70] = 0.75 # weed patch
    msavi_array_input[80:100, 10:30] = 0.3 # grass patch
    msavi_array_input += np.random.rand(120,120) * 0.1 # add some noise

    main_process_msavi_for_weeds(
        msavi_source=msavi_array_input,   # NumPy array
        threshold=0.15,
        output_dir="output_plots",
        file_prefix="numpy_sample1",
        k_clusters=3
    )
    
    # --- Example 3: Case where no significant "weeds" might be found above threshold ---
    low_contrast_msavi = np.random.rand(50,50) * 0.15 # All values below common thresholds
    main_process_msavi_for_weeds(
        msavi_source=low_contrast_msavi,
        threshold=0.25, # Higher threshold, likely no pixels for clustering
        output_dir="output_plots",
        file_prefix="low_contrast_sample",
        k_clusters=3
    )
    
    # Clean up dummy file
    # dummy_tif_path.unlink(missing_ok=True)
    # print(f"Dummy TIF {dummy_tif_path} removed.")