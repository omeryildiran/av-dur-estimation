    # bin_edges = np.histogram_bin_edges(x_data, bins=3)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) // 2
            # y_data_binned = np.zeros(len(bin_centers))
            # bin_sizes = np.zeros(len(bin_centers))
            # for m in range(len(bin_centers)):
            #     bin_mask = (x_data >= bin_edges[m]) & (x_data < bin_edges[m + 1])
            #     y_data_binned[m] = y_data[bin_mask].mean()
            #     bin_sizes[m] = bin_mask.sum()
            # y_data_smooth_binned = pd.Series(y_data_binned)
            
            # # Normalize bin sizes for scatter plot
            # bin_sizes_normalized = bin_sizes / bin_sizes.max() * 100
            ax = axs[idx_data // 2, idx_data % 2]
            # ax.scatter(bin_centers, y_data_smooth_binned, s=bin_sizes_normalized, alpha=0.3)

            #ax.scatter(bin_centers, y_data_smooth_binned, alpha=0.3)