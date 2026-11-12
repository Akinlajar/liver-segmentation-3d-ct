
import matplotlib.pyplot as plt
from ipywidgets import interact


def scroll_3d_arr(data, voxel_size=(1, 1, 1), cmap="gray", dim=0):
    def plot_fn(slice_idx):
        plt.figure(figsize=(7, 7))
        if dim == 0:
            plt.imshow(data[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], cmap=cmap)
        elif dim == 1:
            plt.imshow(data[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], cmap=cmap)
        elif dim == 2:
            plt.imshow(data[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], cmap=cmap)
        else:
            raise ValueError("Invalid dimension. Choose dim=0, dim=1, or dim=2.")

        plt.title(f"Slice {slice_idx} in dimension {dim}")
        plt.axis("off")
        plt.show()

    interact(plot_fn, slice_idx=(0, data.shape[dim] - 1))



def scroll_3d_arr_with_mask(data, mask, voxel_size=(1, 1, 1), cmap="gray", dim=0, alpha=0.2):
    def plot_fn(slice_idx):
        plt.figure(figsize=(7, 7))
        if dim == 0:
            plt.imshow(data[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], cmap=cmap)
            plt.imshow(mask[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], alpha= alpha*(mask[slice_idx, :, :] > 0))
        elif dim == 1:
            plt.imshow(data[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], cmap=cmap)
            plt.imshow(mask[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], alpha= alpha*(mask[:, slice_idx, :] > 0))
        elif dim == 2:
            plt.imshow(data[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], cmap=cmap)
            plt.imshow(mask[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], alpha= alpha*(mask[:, :, slice_idx] > 0))
        else:
            raise ValueError("Invalid dimension. Choose dim=0, dim=1, or dim=2.")

        plt.title(f"Slice {slice_idx} in dimension {dim}")
        plt.axis("off")
        plt.show()

    interact(plot_fn, slice_idx=(0, data.shape[dim] - 1))



def scroll_3d_arr_with_mask(data, mask, voxel_size=(1, 1, 1), cmap="gray", dim=0, alpha=0.2):
    def plot_fn(slice_idx):
        plt.figure(figsize=(7, 7))
        if dim == 0:
            plt.imshow(data[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], cmap=cmap)
            plt.imshow(mask[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], alpha= alpha*(mask[slice_idx, :, :] > 0))
        elif dim == 1:
            plt.imshow(data[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], cmap=cmap)
            plt.imshow(mask[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], alpha= alpha*(mask[:, slice_idx, :] > 0))
        elif dim == 2:
            plt.imshow(data[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], cmap=cmap)
            plt.imshow(mask[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], alpha= alpha*(mask[:, :, slice_idx] > 0))
        else:
            raise ValueError("Invalid dimension. Choose dim=0, dim=1, or dim=2.")

        plt.title(f"Slice {slice_idx} in dimension {dim}")
        plt.axis("off")
        plt.show()

    interact(plot_fn, slice_idx=(0, data.shape[dim] - 1))



def scroll_slices_with_mask_two_plots(data, mask_gt, mask_pred, voxel_size=(1, 1, 1), cmap="gray", dim=0, alpha=0.3):
    def plot_fn(slice_idx):
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        if dim == 0:
            axes[0].imshow(data[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], cmap=cmap)
            axes[0].imshow(mask_gt[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], alpha= alpha*(mask_gt[slice_idx, :, :] > 0))

            axes[1].imshow(data[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], cmap=cmap)
            axes[1].imshow(mask_pred[slice_idx, :, :], aspect=voxel_size[1]/voxel_size[2], alpha= alpha*(mask_pred[slice_idx, :, :] > 0))

        elif dim == 1:
            axes[0].imshow(data[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], cmap=cmap)
            axes[0].imshow(mask_gt[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], alpha= alpha*(mask_gt[:, slice_idx, :] > 0))

            axes[1].imshow(data[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], cmap=cmap)
            axes[1].imshow(mask_pred[:, slice_idx, :], aspect=voxel_size[0]/voxel_size[2], alpha= alpha*(mask_pred[:, slice_idx, :] > 0))

        elif dim == 2:
            axes[0].imshow(data[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], cmap=cmap)
            axes[0].imshow(mask_gt[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], alpha= alpha*(mask_gt[:, :, slice_idx] > 0))

            axes[1].imshow(data[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], cmap=cmap)
            axes[1].imshow(mask_pred[:, :, slice_idx], aspect=voxel_size[0]/voxel_size[1], alpha= alpha*(mask_pred[:, :, slice_idx] > 0))
        else:
            raise ValueError("Invalid dimension. Choose dim=0, dim=1, or dim=2.")


        axes[0].set_title(f"Ground-truth (Slice {slice_idx} in dimension {dim})")
        axes[1].set_title(f"Prediction (Slice {slice_idx} in dimension {dim})")

        axes[0].axis("off")
        axes[1].axis("off")

        plt.show()

    interact(plot_fn, slice_idx=(0, data.shape[dim] - 1))
