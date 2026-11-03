# Dataset folder

Place your public NASA Earthdata dataset here.

Recommended structure:

```text
forest_fire_ai/
  dataset/
    train/
      fire/
      smoke/
      no_fire/
      cloud/
      fog/
      dust/
    val/
      fire/
      smoke/
      no_fire/
      cloud/
      fog/
      dust/
    test/
      fire/
      smoke/
      no_fire/
      cloud/
      fog/
      dust/
```

Official source:
https://www.earthdata.nasa.gov/

Use public MODIS and VIIRS datasets if available. The project will run in demo mode even before the dataset is downloaded.

## Included MODIS data

The project also includes the extracted `modis/2024/` collection from
`C:\ai hackathon\data_set.zip`. It contains 207 country-level CSV files and
`AttributesReadme.txt`. Each CSV uses the NASA FIRMS/MODIS fire-detection
columns, including latitude, longitude, acquisition date/time, confidence,
FRP, and satellite metadata.

This collection is satellite tabular data; it is not an image classification
dataset and should not be moved into the `train/`, `val/`, or `test/` image
class folders.
