import modal

app = modal.App("inspect-aksharantar")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "huggingface_hub"
)


@app.function(image=image, timeout=60 * 10)
def inspect_dataset():
    from datasets import get_dataset_config_names, load_dataset

    ds_name = "ai4bharat/Aksharantar"
    configs = get_dataset_config_names(ds_name)
    print("configs:", configs)

    cfg = configs[0] if configs else None
    if cfg is None:
        print("No configs found")
        return

    ds = load_dataset(ds_name, cfg, split="train[:200]")
    print("columns:", ds.column_names)
    print("num rows sample:", len(ds))

    shown = 0
    for row in ds:
        if shown >= 10:
            break
        print(row)
        shown += 1


@app.local_entrypoint()
def main():
    inspect_dataset.remote()
