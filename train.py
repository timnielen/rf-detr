from rfdetr import RFDETRNano

if __name__ == "__main__":
    model = RFDETRNano()

    model.train(
        dataset_dir="../playdarts/dataset",
        epochs=15,
        batch_size=20,
        grad_accum_steps=1,
        lr=1e-4,
        output_dir="output2",
        wandb=False,
        num_workers=0,
        # resume="output2/checkpoint.pth",
    )