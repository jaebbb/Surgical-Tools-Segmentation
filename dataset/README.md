## Prepare datasets

It is recommended to dataset root to `$NIA-Tool-Segmentation/dataset/input`.  
If your folder structure is different, you may need to change the corresponding paths in files.

```
NIA-Tool-Segmentation
├── dataset
│   ├── input
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├── .....
│   │   ├── .....
│   │   ├── image1.json
│   │   ├── image2.json
│   │   ├── .....
│   │   ├── .....
```  

### Json Format  
```
{
    "version": "xx.xx.xx",
    "flags" : {},
    "shapes" : [
        {
            "label": "1",
            "points": [ [,] , [,] ],
            "group_id": null,
            "shape_type": "polygon",
            "inner": null,
            "flags": {}
        }, 
        .
        .
        .
        .
    ]
    "imagePath": "image{NUM}.jpg",
    "imageData": " ",
    "imageHeight": height,
    "imageWidth": width
}
```

### Run  
```
sh preprocessing.sh
```
