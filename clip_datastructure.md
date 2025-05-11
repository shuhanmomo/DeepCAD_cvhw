clip feats path: "data/CLIP_feats.json"

clip json structure:
```
{
    "train": [
        {
            "id": "0000/00001926",
            "clip_feats": [
                [feat1],  // CLIP feature for view 1
                [feat2],  // CLIP feature for view 2
                ...
                [feat24]  // CLIP feature for view 24
            ]
        },
        ...
    ],
    "validation": [...],
    "test": [...]
}
```
