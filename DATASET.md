### PartNet Dataset: 
The Partnet Dataset is tructured in the following manner:


```
PartNet/
├── data_v0/
│   ├── 1/
│   │   ├── obj/
│   │   │   ├── new-0.obj
│   │   │   └── original-1.obj
│   │   ├── parts_render/
│   │   │   ├── 0.png
│   │   │   └── 0.txt
│   │   ├── parts_render_after_merging/
│   │   │   ├── 0.png
│   │   │   └── 0.txt
│   │   ├── point_sample/
│   │   │   ├── label-10000.txt
│   │   │   ├── pts-10000.txt
│   │   │   ├── pts-10000.ply
│   │   │   ├── pts-10000.pts
│   │   │   ├── sample-points-all-label-10000.txt
│   │   │   ├── sample-points-all-pts-label-10000.ply
│   │   │   ├── sample-points-all-pts-nor-rgba-10000.ply
│   │   │   └── sample-points-all-pts-nor-rgba-10000.txt
│   │   ├── meta.json
│   │   ├── result_after_merging.json
│   │   ├── result.json
│   │   ├── tree_hier_after_merging.html
│   │   └── tree_hier.html
│   ...
└── 26671/
    ...

```
Download: You can download the processed data from [PartNet Dataset](https://github.com/daerduoCarey/partnet_dataset) repository, or download from the [official website](https://shapenet.org/download/parts) and process it by yourself.