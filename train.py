import modal

app = modal.App(name="train-rsna")

DATA_DIR = "/data"
OUTPUT_DIR = "/working"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "torchvision",
        "pandas",
        "albumentations",
        "scikit-learn",
        # "git+https://github.com/qimaotaolun/pytorch-image-models.git",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        # force_build=True,
        ).apt_install("git").run_commands(
        "cd root && git clone https://github.com/qimaotaolun/pytorch-image-models.git",
        "cd root && pip install -q -e pytorch-image-models",
        # force_build=True,
        )

volume = modal.Volume.from_name(
    "tmp_dataset",
)
output_volume = modal.Volume.from_name(
    "tmp_output", create_if_missing=True
)

@app.function(
    volumes={DATA_DIR: volume, OUTPUT_DIR: output_volume},
    image=image,
    timeout=60*60*24,  # 24 hours
    gpu="T4:1",
    
)
def train():
    import subprocess

    def _exec_subprocess(cmd: list[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    print("launching training script")
 
    _exec_subprocess(
        [
            f"bash {DATA_DIR}/rsna-data/distributed_train.sh 1",
        ]
    )        
    volume.commit()