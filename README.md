# 3D Shape Reconstruction

Given an RGB image of an object, a sequence of tactile readings from the object’s surface, or a sequence of impact sounds of striking its surface locations, the task is to reconstruct the point cloud of the target object given combinations of these multisensory observations.

## Usage

#### Data Preparation

The dataset used to train the baseline models can be downloaded from [here](https://www.dropbox.com/scl/fo/buy9lng0hlmk36rk0gfev/AJw2TP0KA_IfhxrQWIRsXUU?rlkey=3gps10mku9lmp4t2i9uayqqk5&st=wfur8rt3&dl=0)

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
# Train PCN as an example
python main.py --modality_list vision touch audio \
							 --model PCN \
							 --batch_size 8 \
               --epochs 10 \
               --local_gt_points_location ../DATA_new/local_gt_points_down_sampled \
               --lr 1e-4 --exp pcn/vision_touch_audio \
               --config_location ./configs/PCN.yml \
               --normalize
```

Evaluate the best model in *pcn/vision_touch_audio*:

```sh
# Evaluate PCN as an example
python main.py --modality_list vision touch audio \
							 --model PCN \
							 --batch_size 8 \
               --epochs 10 \
               --local_gt_points_location ../DATA_new/local_gt_points_down_sampled \
               --lr 1e-4 --exp pcn/vision_touch_audio \
               --config_location ./configs/PCN.yml \
               --normalize --eval
```

#### Add your own model

To train and test your new model on ObjectFolder 3D Shape Reconstruction Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

   ```sh
   mkdir models/my_model
   ```

2. Design new model

   ```sh
   cd models/my_model
   touch my_model.py
   ```

3. Build the new model and its optimizer

   Add the following code into *models/build.py*:

   ```python
   elif args.model == 'my_model':
       from my_model import my_model
       model = my_model.my_model(args)
       optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   ```

4. Add the new model into the pipeline

   Once the new model is built, it can be trained and evaluated similarly:

   ```sh
   python main.py --modality_list vision touch audio \
   							 --model my_model \
   							 --batch_size 8 \
                  --epochs 10 \
                  --local_gt_points_location ../DATA_new/local_gt_points_down_sampled \
                  --lr 1e-4 --exp my_model/vision_touch_audio \
                  --config_location ./configs/my_model.yml \
                  --normalize
   ```

## Results on ObjectFolder 3D Reconstruction Benchmark

For the visual RGB images, tactile RGB images, and impact sounds used in this task, we respectively sample 100 instances around each object (vision) or on its surface (touch and audio).

In all, given the 1, 000 objects, we can obtain 1, 000 x 100 = 100, 000 instances for vision, touch, and audio modality, respectively. In the experiments, we randomly split the 1, 000 objects as train/validation/test = 800/100/100, meaning that the models need to generalize to new objects during testing. Furthermore, we also test the model performance on ObjectFolder Real by similarly splitting the 100 objects as train/validation/test = 60/20/20.

#### Results on ObjectFolder

<table>
    <tr>
        <td>Method</td>
        <td>Vision</td>
        <td>Touch</td>
        <td>Audio</td>
        <td>V+T</td>
        <td>V+A</td>
        <td>T+A</td>
        <td>V+T+A</td>
    </tr>
    <tr>
        <td>MDN</td>
        <td>4.02</td>
        <td>3.88</td>
        <td>5.04</td>
        <td>3.19</td>
        <td>4.05</td>
        <td>3.49</td>
        <td>2.91</td>
    </tr>
  <tr>
        <td>PCN</td>
        <td>2.36</td>
        <td>3.81</td>
        <td>3.85</td>
        <td>2.30</td>
        <td>2.48</td>
        <td>3.27</td>
        <td>2.25</td>
    </tr>
  <tr>
        <td>MRT</td>
        <td>2.80</td>
        <td>4.12</td>
        <td>5.01</td>
        <td>2.78</td>
        <td>3.13</td>
        <td>4.28</td>
        <td>3.08</td>
    </tr>
</table>

#### Results on ObjectFolder Real

<table>
    <tr>
        <td>Method</td>
        <td>Vision</td>
        <td>Touch</td>
        <td>Audio</td>
        <td>V+T</td>
        <td>V+A</td>
        <td>T+A</td>
        <td>V+T+A</td>
    </tr>
    <tr>
        <td>MRT</td>
        <td>1.17</td>
        <td>1.04</td>
        <td>1.04</td>
        <td>0.96</td>
        <td>1.50</td>
        <td>1.12</td>
        <td>0.95</td>
    </tr>
</table>
