# YCB mesh dataset.

## How we downloaded the dataset.

Following recommendations on https://www.ycbbenchmarks.com/object-models,
https://github.com/sea-bass/ycb-tools downloader was used to fetch the dataset
to a local machine.

```shell
git clone https://github.com/sea-bass/ycb-tools .
python3 download_ycb_dataset.py
```

The original list of objects could be found in this
[JSON](https://ycb-benchmarks.s3.amazonaws.com/data/objects.json). We believe
"027-skillet" in this file is misspelled, it should be "027_skillet". The data
was fetched manually by creating the correct
[URL for "skillet"](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/027_skillet_google_16k.tgz).

We use google meshes with 16k vertices from `google_16` directory, but a few of
the models in the dataset do not have 'google_16k' meshes available. These
objects will be further examined to make them available in simulation.
