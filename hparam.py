from network import AE_MNIST
import argparse
from optim import optimizer as op
from loss import loss as LOSS
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from data import NPZ_Dataset
from Metrics import cluster_metrics as CM
import utils
from collections import defaultdict
import sys
sys.path.insert(0, "/home/beast/BEAST/Tejas/DOD_CLUSTERING/")
warnings.simplefilter(action='ignore', category=FutureWarning)



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def cluster(model, loader, device):
    model.eval()
    z = []
    Y = []
    for (x, y, _) in loader:
        x = x.to(device)
        _, _, Z = model(x)
        z.extend(Z.detach().cpu().numpy())
        Y.extend(y.numpy())

    return np.array(z), np.array(Y)




def MEAN(x):
    return sum(x)/len(x)


def get_dict(glob, loc):
    if glob == None:
        glob = defaultdict(list)
        for key in loc.keys():
            glob[key] = [loc[key]]
    else:
        for key in loc.keys():
            glob[key].append(loc[key])

    return glob


def get_sample(path, device):
    f = np.load(path)
    x, y, = f['x_test'], f['y_test']

    return torch.tensor(x).to(device)


def save_best(model, acc, results, best_acc, epoch, optimizer, name, z, Y):
    if best_acc == None:
        best_acc = acc
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'acc': acc}
        results.save_model(dict, name)
        results.save_latent("best", z, Y, name='Z')

        print("saved", best_acc)

    elif acc >= best_acc:
        best_acc = acc
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'nmi': acc}

        results.save_model(dict, name)
        print("saved", best_acc)

    return best_acc


def main(**kwargs):

    for batch_size in kwargs["batch_sizes"]:
        for lr in kwargs["learning_rates"]:
            for optimizer in kwargs["optimizers"]:
                for alpha in kwargs["alphas"]:
                    for dod in kwargs["dod"]:
                        for beta in  kwargs["betas"]:

                            eval_dict1 = None
                            r_path = kwargs["root"] + \
                                f"dataset:{kwargs['dataset']}_BS:{batch_size}_LR:{lr}_optim:{optimizer}_alpha:{alpha}_beta:{beta}_dod_dis:{dod}recentre:{kwargs['recenter']}"

                            results = utils.save_results(r_path)

                            device = utils.set_seed_globally(
                                seed_value=0, if_cuda=True, gpu=kwargs['gpu'])
                            dataset = NPZ_Dataset(str=kwargs["dataset"])

                            model = AE_MNIST(dod).to(device)
                            print(model)

                            pretrain_loader = DataLoader(
                                dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=kwargs["num_workers"])

                            cluster_loader = DataLoader(
                                dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=kwargs["num_workers"])
                            criterion_mse = LOSS(device, loss='mse')
                            criterion_ssim = LOSS(device, loss='ssim', channel=1)
                            optim = op(model, lr, optimizer)
                            Optim = optim.call()

                            MSE = []
                            SSIM = []
                            RECON = []
                            DOD_LOSS = []

                            for epoch in range(kwargs["epochs"]):
                                loop = tqdm(enumerate(pretrain_loader), total=len(
                                    pretrain_loader), leave=False, colour='green')
                                model.train()

                                mse = []
                                ssim = []
                                recon = []
                                dod_loss = []

                                for idx, (x, _, _) in loop:
                                    x = x.to(device)
                                    x_bar, a, z = model(x)

                                    Optim.zero_grad()

                                    mse_loss = criterion_mse(x_bar, x)
                                    ssim_loss = 1 - criterion_ssim(x_bar, x)
                                    recon_loss = alpha*mse_loss + \
                                        (1-alpha)*ssim_loss

                                    dodloss = model.dod_loss(x,a,z)

                                    loss = (1-beta)*recon_loss + beta*dodloss
                                    loss.backward()
                                    Optim.step()

                                    mse.append(mse_loss.item())
                                    ssim.append(ssim_loss.item())
                                    recon.append(recon_loss.item())
                                    dod_loss.append(dodloss.item())

                                    if idx % kwargs["show"] == 0:
                                        E = kwargs["epochs"]
                                        loop.set_description(f"[{epoch}/{E}]")
                                        loop.set_postfix(Recon_loss=recon_loss.item(
                                        ), dod_loss=dodloss.item())

                                x_sample = get_sample(kwargs["sample_path"],device)
                                with torch.no_grad():
                                    model.eval()
                                    x_hat,_,_ = model(x_sample)
                                    results.save_images(epoch,x_sample,x_hat)
                                model.train()

                                MSE.append(MEAN(mse))
                                SSIM.append(MEAN(ssim))
                                RECON.append(MEAN(recon))
                                DOD_LOSS.append(MEAN(dod_loss))

                                loss_frame = {"MSE": MSE, "SSIM": SSIM,
                                            "RECON": RECON, "DOD_LOSS": DOD_LOSS}
                                results.save_loss(loss_frame)

                                if (epoch+1) % kwargs["recenter"] == 0:
                                    # X = torch.tensor(dataset.x)
                                    Y = []
                                    z, Y = cluster(model, cluster_loader, device)

                                    z_metrics = CM.all_metrics(
                                        latent=z, y=Y, n_clusters=kwargs["clusters"], n_init=20, n_jobs=-1)

                                    z_scores = z_metrics.scores()

                                    eval_dict1 = get_dict(eval_dict1, z_scores)

                                    results.save_eval_metric(eval_dict1, name='Z')

                                    print(
                                        f"z acc:{z_scores['acc']:.4f},nmi:{z_scores['nmi']:.4f}")

                                    acc = z_scores['acc']

                                    dict = {'weights': model.state_dict(),
                                            'optimizer': Optim.state_dict(),
                                            'epoch': epoch,
                                            'acc': acc}

                                    results.save_model(dict, f"LastEpoch")
                                    if (epoch+1) == kwargs["recenter"]:
                                        best_acc = None

                                    best_acc = save_best(
                                        model, acc, results, best_acc, epoch, Optim, "best_z1", z, Y)

                                    model.train()


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="LST")
    args.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="MSE*alpha + (1-aplha)*SSIM"

    )
    c = args.parse_args()
    main(
        root="/home/beast/BEAST/Tejas/DOD_CLUSTERING/FMNIST/logs/",
        sample_path='/home/beast/BEAST/DATA/DATASET_NPZ/FMNIST10.npz',
        epochs=200,
        batch_sizes=[512],
        learning_rates=[1e-3],
        optimizers=["adam"],
        alphas=[c.alpha],
        betas = [0.25,0.5,0.75],
        dataset="FMNIST",
        show=10,
        recenter=10,  # num epochs before dataset clustering
        clusters=10,
        gpu=0,
        num_workers =6,
        dod=["mse","mae","cosine"]
    )
