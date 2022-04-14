from torch.utils.data import DataLoader, random_split

from inference.train import train

from simulate.simulate import simulate

SEED_DATA_GENERATION = 0


def main():

    dataset, truth = simulate(use_torch=True, seed=SEED_DATA_GENERATION,
                              use_torch_dataset=True)
    n = len(dataset)

    prop_training = 0.8
    n_training = int(prop_training*n)
    n_testing = n - n_training

    train_set, val_set = random_split(
        dataset,
        [n_training, n_testing])

    print("N training", n_training)
    print("N testing", n_testing)

    batch_size = n

    training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=n_testing, shuffle=True)

    z_flow, theta_flow, hist_loss, hist_loss_val = train(

        truth=truth,
        load_bkp=False,
        bkp_name="norm_flow_split",
        epochs=0)

    # # for d in val_data:
    # log_loss = loss_val(
    #     n_u=dataset.n_u,
    #     n_w=dataset.n_w,
    #     z_flow=z_flow,
    #     n_sample=100,
    #     **val_data)

    # print(log_loss.item())

    # loss = torch.nn.BCELoss()
    # print(loss(torch.ones(n_testing, dtype=torch.double)*0.5, val_data['y'].squeeze()))
    # 0.6931







if __name__ == "__main__":
    main()
