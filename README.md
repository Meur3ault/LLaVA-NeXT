#Nano LLaVA OneVision Practice

  Hi, everyone, this practics is going to guide you to train a LLaVA OneVision with the ViT-L encoder and Qwen-2 0.5B model. There is two part including:


1.   The step-by-step training of your own LLaVA-onevision model.
2.   The evaluation and inference of the given model.

Or you could just used my completed checkpoint in https://huggingface.co/RuaZhou/Nano_LLaMA_Onevision for inference & evaluation.

<img width="728" height="404" alt="image" src="https://github.com/user-attachments/assets/8bda9609-f9fc-4946-9181-3eb271cb653e" />


The whole training process will take time near a day with the RTX 4070 Ti level graphical card. **I modified [the original LLaVA-Next code](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main)** a little bit so that it could be trained on **16GB-consummer-level** card. For me, it was implemented with RTX 5060ti.

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

(You have to upgrade your torch version first)
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

## 1.2 Stage-1: Pre-training of projector

Firstly, the data set for pre-training stage should be prepared by downloading the [blip_558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) data set. Then the **image.zip and json data** should be unzipped and placed with the correspoding images and data path as **LLaVA-NeXT/scripts/train/pretrain_siglip.sh** indicates:

<img width="625" height="241" alt="image" src="https://github.com/user-attachments/assets/419d515e-9c90-4a9c-8986-6a7f8c86037a" />

Then, under the main path, run the command:


```
./scripts/train/pretrain_siglip.sh
```
If it is inexecutable, please change it into executable file:
```
chmod +x ./scripts/train/pretrain_siglip.sh
```
The pretraining stage will begin to train the projector for aligning the modalities between vision tokens and languages tokens.

<img width="1710" height="832" alt="image" src="https://github.com/user-attachments/assets/32aa3dc0-9be2-4c86-891b-777bc703e0eb" />

## 1.3 Stage-1.5: Mid-stage training

The mid-stage data set [synthdog_en](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/synthdog_en) should be downloaded [here](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main/synthdog_en) and be put into the path order as the figure shows below:

<img width="1177" height="397" alt="image" src="https://github.com/user-attachments/assets/0c49b37a-38d8-447c-867d-50d35dba74c0" />

Then under the root path of this project, run the following command to training the stage 1.5:


```
./scripts/train/mid_stage.sh
```
And you could modify the code below. Especially if you haven't installed flash attention 2, you shall change the --attn_implementation from **flash_attention_2** back to torch-based **sdpa**.


<img width="580" height="851" alt="image" src="https://github.com/user-attachments/assets/b7122bff-a0e1-43f8-a26a-c512501f48e5" />


## 1.4 Stage-2: Single image training

After finishing the stage 1.5, i.e., mid stage, the following [single-image data sets](https://huggingface.co/datasets/jylins/LLaVA-OneVision/tree/main) ( images & jsons ) need to be downloaded and be ordered as below:

<img width="1172" height="954" alt="image" src="https://github.com/user-attachments/assets/9fba0cbe-79c7-4e23-a1d1-71bf16cb7fdf" />
<img width="769" height="184" alt="image" src="https://github.com/user-attachments/assets/4a7518c7-de46-4c82-96b0-c27bdd625d45" />


The selected dataset includes: 


1.   ai2d_gpt4v
2.   ai2d_internvl
3.   ai2d_llava_format
4. chart2text
5. chartqa
6. dvqa
7. image_textualization
8. infographic_vqa
9. infographic_vqa_gpt4v
10. infographic_vqa_llava_format

The first **20%** percent of the selected data set is used, near **73K** samples.
If you want to change the training content, please modify the file **LLaVA-NeXT/scripts/train/nano_single_image.yaml** .

And its configurations have been modified to fit in 16GB GPU device (so as the mid-stage training configs and OneVision training configs):

1.   Image_aspect_ratio has been shrinked. The **anyres_max_#** basically means that it allows the basic brief tokens ×729 from the resized 384×384 figure andadditional **#×729 tokens** from **AnyRes** while every 384×384 split is generating 729 tokens. Once it is surpassed, the additional tokens from **AnyRes** will be bilinear interpolated for downsampling untile the given size.

2.   The learning rate and batch size have been reduced.

3.   The image_grid_pinpoints has been modifed. It means what the model would choose the closest resizing resolution for the given image. For example, for [384, 600], the model will choose [384, 768] to split the image for **AnyRes**. Here the options are restricted into "[[384,384],[384,768],[768,384]]".

4. The tunable parts have been limited within the projector and the LLM backbone.

<img width="799" height="403" alt="image" src="https://github.com/user-attachments/assets/e33fa2e6-e547-4e2f-8107-3c48afbdbe5c" />

Run the following code to execute the script:

```
./scripts/train/finetune_si.sh
```

## 1.5 Stage-2: OneVision training
After finishing the stage 2 of single images, the selected multi-image set and video set need to be downloaded from [multi-image data sets](https://huggingface.co/datasets/jylins/LLaVA-OneVision/tree/main) and  [lmms-lab/LLaVA-Video-178K/30_60_s_nextqa](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/30_60_s_nextqa). They should be ordered as below.


The selected multi-image data sets include: 


1.   DocVQA
2.   OCR-VQA
3.   RAVEN_train_images
4.   Spot-the-Diff

The video data set includes:

1.   NextQA


Finally, the customized data set inlcudes near 49.5K multi-image samples and near 1.7k video samples. The first 20% samples were used, corresponding to 9.9K multi-images data and 0.34K video data. If you want to change the training content, please modify the file **LLaVA-NeXT/scripts/train/nano_onevision.yaml** .


<img width="809" height="294" alt="image" src="https://github.com/user-attachments/assets/4d3575d4-2ec1-4d14-a3db-117d8c6b0b63" />

<img width="945" height="505" alt="image" src="https://github.com/user-attachments/assets/508e1083-b7fd-493c-ade1-43c85dea5e96" />

<img width="723" height="208" alt="image" src="https://github.com/user-attachments/assets/17770463-3a33-426a-9d98-7a978cc70a49" />

<img width="835" height="275" alt="image" src="https://github.com/user-attachments/assets/ef443f1d-b61d-449a-a4ed-ff7223571c3d" />

Beside of the modification in other stages, the OneVision stage further reduce the **image_grid_pinpoints**, **image_aspect_ratio**, and **frame_upbound**. Meanwhile, the model_max_length is increased to ensure processing multi-image tokens.

<img width="595" height="689" alt="image" src="https://github.com/user-attachments/assets/cd1aff52-b8c3-440c-98fd-a4a1a56969c1" />

Run the following command to complete the training:


```
./scripts/train/finetune_ov.sh
```

# 2 Inference & Evaluation

# 2.1 Inference for images and videos

Install the jupyter if the environment does not have:
```
pip install jupyter
```
Run the code of ./Test.ipynb

The single-image tuning model is much more stable then OneVision tuning, which may attribute to the data set quality and coarse config in ./scripts/train/finetune_ov.sh .

<img width="910" height="380" alt="image" src="https://github.com/user-attachments/assets/920cc108-ce04-47cf-9ab9-63a2739fa375" />


# 2.2 Evaluation

To evaluate the trained model, you may need a huggingface token to log in to download the evaluation data set.



```
huggingface-cli login
```

Installation:

cd to the root dir of the project:



```
cd ./LLaVA-NeXT
```

Install lmm-eval for evaluation:


```
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval

cd lmms-eval

```

Run the following command for evaluation, if error for import packages, just run **pip install xxxxxx** to fix.

And the evaluation sets **flash attention 2** as the default options, if not, you may modify it a lot to evaluate.

**If you want to evaluate the OneVision checkpoint trained by this repository (the huggingface checkpoint is already revised)**, please replace the config of **LLaVA-NeXT/checkpoints/onevision/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-ov_stage_am9/config.json** (or the config of your checkpoints) from:

<img width="457" height="158" alt="image" src="https://github.com/user-attachments/assets/e668b891-fbcd-4c2a-a33b-40adc14d26be" />

to:

<img width="441" height="312" alt="image" src="https://github.com/user-attachments/assets/a26cc54b-f46b-45f4-9e0d-996cf42ceaa8" />

to avoid the zero-division error. 



**1.   Single image bench mark**



```
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=../checkpoints/onevision/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-ov_stage_am9,conv_template=qwen_1_5,device_map=cuda,model_name=llava_qwen \
    --tasks=ai2d,chartqa,docvqa_val,mmmu_pro \
    --batch_size=1
```
You can substitute the model with my already trained checkpoint:

```
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=RuaZhou/Nano_LLaMA_Onevision,conv_template=qwen_1_5,device_map=cuda,model_name=llava_qwen \
    --tasks=ai2d,chartqa,docvqa_val,mmmu_pro \
    --batch_size=1
```



**2.   Video bench mark**



```
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=../checkpoints/onevision/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-0.5B-Instruct-ov_stage_am9,conv_template=qwen_1_5,device_map=cuda,model_name=llava_qwen \
    --tasks=seedbench,ocrbench \
    --batch_size=1
```
You can substitute the model with my already trained checkpoint:

```
accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=RuaZhou/Nano_LLaMA_Onevision,conv_template=qwen_1_5,device_map=cuda,model_name=llava_qwen \
    --tasks=seedbench,ocrbench \
    --batch_size=1
```

The result is 

<img width="1092" height="564" alt="image" src="https://github.com/user-attachments/assets/aa30b767-2ca7-4964-83b0-cce1142ed1e9" />

Whereas, the original result of LLaVA OneVision 0.5B, is:


Images:
1.   AI2D : 57.1% (ours is 46.86%)
2.   ChartQA: 61.4% (ours is 7.88%)
3.   DocVQA_val: 70.0% (ours is 15.90%)
4.   SeedBench (image):  65.5% (ours is 49.18%)

Videos:
1.   SeedBench (video):  44.2% (ours is 42.03%)




