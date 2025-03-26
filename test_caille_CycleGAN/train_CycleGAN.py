import os
import time
from options.train_options import TrainOptions
from util.visualizer import Visualizer  # À créer plus tard si tu veux de l'affichage
from test_caille_CycleGAN.h5_dataset import H5UnalignedDataset
from test_caille_CycleGAN.Cycle_GAN_class import MultiStainCycleGANModel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize

def main():
    opt = TrainOptions().parse()  # Récupère les options (ex: batch_size, lr, etc.)
    print(f"[INFO] Options : {opt}")

    # Dataset : train = domaine A, val = domaine B
    transform = Compose([
        Resize((100, 100))  # ça garde [3, 98, 98]
        ])



    dataset = H5UnalignedDataset(
        h5_path_A='data/train.h5',
        h5_path_B='data/val.h5',
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    print(f"[INFO] Nombre d'images d'entraînement : {len(dataset)}")

    # Création et setup du modèle CycleGAN
    model = MultiStainCycleGANModel(opt)
    model.setup(opt)

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        print(f"\n🔁 Début époque {epoch}")

        #model.train()
        model.isTrain = True
        for i, data in enumerate(dataloader):
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.loss_G, model.loss_D_A, model.loss_D_B
                print(f"[Epoch {epoch} | Iter {epoch_iter}] Losses G: {losses[0]:.4f}, D_A: {losses[1]:.4f}, D_B: {losses[2]:.4f}")

        if epoch % opt.save_epoch_freq == 0:
            print(f"[💾] Sauvegarde à l'époque {epoch}")
            model.save_networks(epoch)

        print(f"[✅] Fin époque {epoch} | Temps écoulé : {time.time() - epoch_start_time:.2f}s")
        model.update_learning_rate()

if __name__ == '__main__':
    main()
