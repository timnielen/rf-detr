from rfdetr import RFDETRNano

model = RFDETRNano()

model.train(
    dataset_dir="../playdarts/dataset",
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output_model",
    wandb=False,
    project="RFDETR",
    run="train_0",
    world_size=1,
    num_workers=0,
)