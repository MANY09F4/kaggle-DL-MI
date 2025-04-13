import os
import time
from test_caille_CycleGAN.options.train_options import TrainOptions
from test_caille_CycleGAN.util.visualizer import Visualizer
from test_caille_CycleGAN.h5_dataset import H5UnalignedDataset
from test_caille_CycleGAN.Cycle_GAN_class import MultiStainCycleGANModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize

def main():
    opt = TrainOptions().parse()  # Récupère les options (ex: batch_size, lr, etc.)

    print(f"[INFO] Options : {opt}")

    transform = None  # Pas de transformation si tu ne veux pas la modifier.

    aberrant_ids_map = {
    opt.train_path: [int(i) for i in opt.aberrant_ids_train.split(",")],
    opt.val_path: [int(i) for i in opt.aberrant_ids_val.split(",")]
        }
    # Convertir aberrant_ids en liste
    #aberrant_ids = [int(id) for id in opt.aberrant_ids.split(',')] if opt.aberrant_ids else []

    # Si domain est spécifié, charger seulement le domaine source sélectionné.
    if opt.domain is None:
        # Si aucun domaine spécifique n'est sélectionné, on prend train et val comme sources et test comme cible
        dataset = H5UnalignedDataset(
            h5_path_A=[opt.train_path, opt.val_path],
            h5_path_B=opt.test_path,
            transform=transform,
            aberrant_ids_map=aberrant_ids_map
        )
    else:
        # Si un domaine source est sélectionné, on charge seulement ce domaine source
        dataset = H5UnalignedDataset(
            h5_path_A=[opt.train_path, opt.val_path],
            h5_path_B=opt.test_path,
            domain=opt.domain,  # Passe le domaine sélectionné
            transform=transform,
            aberrant_ids_map=aberrant_ids_map
        )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    print(f"[INFO] Nombre d'images d'entraînement : {len(dataset)}")

    # Création et setup du modèle CycleGAN
    model = MultiStainCycleGANModel(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)

    total_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate()

        epoch_start_time = time.time()
        epoch_iter = 0
        print(f"\n🔁 Début époque {epoch}")
        visualizer.reset()

        model.isTrain = True
        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(f"[Epoch {epoch} | Iter {epoch_iter}] " + ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch, save_result=True)

        if epoch % opt.save_epoch_freq == 0:
            print(f"[💾] Sauvegarde à l'époque {epoch}")
            model.save_networks(epoch)

        print(f"[✅] Fin époque {epoch} | Temps écoulé : {time.time() - epoch_start_time:.2f}s")

if __name__ == '__main__':
    main()
