## 1st Place of the Kidney Pathology Image segmentation ([KPIs](https://sites.google.com/view/kpis2024/)) challenge - MICCAI 2024 - Team Capybara

### Solution video
Please see the 1st place solution [here](https://drive.google.com/file/d/1kMzpaz8CYoBC1vAf_4NfzoPjBmLKPCyu/view) or check out top-team solutions at the KPIs [leaderboard](https://sites.google.com/view/kpis2024/leaderboard).

### Docker images
Inference code and pre-trained model are provided in docker images at: https://hub.docker.com/r/quancap/kpis24

There are 2 docker images but they have the **same** model weight as we use the same model but for two tasks.  
Trained model weight is located in `/workspace/KPIs_2024/` inside docker images.

#### For patch-level segmentation task
```bash
docker pull quancap/kpis24:task_1_patch
```
Execute command
```bash
docker run --gpus all -it -v /your_input_dir/:/input -v /your_output_dir/:/output quancap/kpis24:task_1_patch /workspace/KPIs_2024/script_task_1.sh
```

#### For WSI-level segmentation task
```bash
docker pull quancap/kpis24:task_2_slide
```
Execute command
```bash
docker run --gpus all -it -v /your_input_dir/:/input -v /your_output_dir/:/output quancap/kpis24:task_2_slide /workspace/KPIs_2024/script_task_2.sh
```

For more details of data structure and preparation, visit KPIs's [evaluation page](https://sites.google.com/view/kpis2024/evaluation).

### To-do
- [ ] Prodivde inference code and model separately from docker images