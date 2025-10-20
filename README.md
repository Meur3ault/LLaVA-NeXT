#Nano LLaVA OneVision Practice

  Hi, everyone, this practics is going to guide you to train a LLaVA OneVision with the ViT-L encoder and Qwen-2 0.5B model. There is two part including:


1.   The step-by-step training of your own LLaVA-onevision model.
2.   The evaluation and inference of the given model.

Or you could just used my completed checkpoint in https://huggingface.co/RuaZhou/Nano_LLaMA_Onevision for inference & evaluation.
![Figure1]

The whole training process will take time near a day with the RTX 4070 Ti level graphical card. **I modified [the code](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main)** a little bit so that it could be trained on **16GB-consummer-level** card. For me, it was implemented with RTX 5060ti.

# 1 Training

The training process includes four part:


1.   Stage-1: Pre-training of projector (**about 21h**)
2.   Stage-1.5: Mid-stage training (**about 3h**)
3.   Stage-2: Single image training (**about 10h**)
4.   Stage-2: OneVision training (**about 6h**)

Then we are going to explain each part of it in details.

And it's strongly recommended that you may use the following command to shutdown graphical interface temporarily while training:
```
sudo systemctl isolate multi-user.target
```

then recovering the interface:


```
sudo systemctl isolate graphical.target
```

This will bring near 12% speed up.

## 1.1 Installation

Firstly, you need to clone this modified repository and navigate to the LLaVA folder:

```
git clone https://github.com/Meur3ault/LLaVA-NeXT.git
cd LLaVA-NeXT
```

Then install the reference packages:

```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install -r requirements.txt
```

If you want to use the **[flash attention packages](https://github.com/Dao-AILab/flash-attention)** for evaluation and speeding up, please make sure the **cuda verison>=12.0**. Then run


```
pip install flash-attn --no-build-isolation
```

If you want to record the training-related data automatically like **the training curves above**, please register an account at [wandb.ai](https://wandb.ai/authorize) for a token , then



1.   Set the WANDB_API_KEY environment variable.

  ```
       export WANDB_API_KEY=<your_api_key>
  ```


2.   Install the wandb library (it is already in the requirement, so you may run wandb login directly) and log in.

  ```
       pip install wandb
       wandb login
  ```

Finally, due to the imcompleted list of requirement, you may need to run the following command or any other command noticing to install **the missing package**:



```
pip install sqlitedict tenacity Levenshtein pytablewriter
```









