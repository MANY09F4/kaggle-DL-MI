import os
import time
from CycleGAN.options.train_options import TrainOptions
from CycleGAN.util.visualizer import Visualizer
from CycleGAN.h5_dataset import H5UnalignedDataset
from CycleGAN.Cycle_GAN_class import MultiStainCycleGANModel
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    # Parse command-line options (batch size, learning rate, etc.)
    opt = TrainOptions().parse()
    print(f"[INFO] Parsed options: {opt}")

    # Define optional image transformations (e.g., Resize, Normalization)
    transform = None

    # Build dictionary with IDs of aberrant images to exclude, per HDF5 file
    aberrant_ids_map = {
        opt.train_path: [int(i) for i in opt.aberrant_ids_train.split(",")],
        opt.val_path: [int(i) for i in opt.aberrant_ids_val.split(",")]
    }

    # Build dataset based on the selected source domain (or multiple if not specified)
    if opt.domain is None:
        dataset = H5UnalignedDataset(
            h5_path_A=[opt.train_path, opt.val_path],
            h5_path_B=opt.test_path,
            transform=transform,
            aberrant_ids_map=aberrant_ids_map
        )
    else:
        dataset = H5UnalignedDataset(
            h5_path_A=[opt.train_path, opt.val_path],
            h5_path_B=opt.test_path,
            transform=transform,
            domain=opt.domain,
            aberrant_ids_map=aberrant_ids_map
        )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    print(f"[INFO] Training samples: {len(dataset)}")

    # Initialize CycleGAN model
    model = MultiStainCycleGANModel(opt)
    model.setup(opt)

    # Set up visualization tool
    visualizer = Visualizer(opt)

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate()
        epoch_start_time = time.time()
        epoch_iter = 0
        print(f"\n[INFO] Starting epoch {epoch}")
        visualizer.reset()
        model.isTrain = True

        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(f"[Epoch {epoch} | Iter {epoch_iter}] " +
                      ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result=True)

        # Save the model at specified frequency
        if epoch % opt.save_epoch_freq == 0:
            print(f"[INFO] Saving model at epoch {epoch}")
            model.save_networks(epoch)

        print(f"[INFO] End of epoch {epoch} | Time elapsed: {time.time() - epoch_start_time:.2f}s")

if __name__ == '__main__':
    main()
