# BeetleFlow: An Integrative Deep Learning Pipeline for Beetle Image Processing

![main figure](assets/pipeline.png)

## 🗓️ TODO
- [ ] Update data source
- [ ] Provide checkpoint downloads
- [ ] Add installation guide

## 📊 Dataset
Beetle part segmentation dataset is available [here](data/) (both 5-class and 9-class).

## 🧑‍💻 Usage

### Detection

```bash
python pipeline/1_detection/detection_beetles.py \
    --image-dir=/path/to/input/tray/image/dir \
    --output-dir=/path/to/output/dir \
    --metadata-file /path/to/all/individual/metadata.csv
```

### Cropping

Modify the paths in `1_sort_and_crop.py`, `2_match_metadata.py`, and `3_check_matching.py`, then run:
```bash
python 1_sort_and_crop.py
python 2_match_metadata.py
python 3_check_matching.py
```

### Segmentation

Download the checkpoints, then run:
```bash
python batch_inference.py \
    --input_dir /path/to/cropped/images/dir \
    --output_dir /path/to/output/dir \
    --model ./checkpoints/5-class # or 9-class
```

## License

Though the code in this repository is shared under an [MIT License](https://github.com/Imageomics/BeetleFlow/blob/main/LICENSE), the data (images) are licensed under [Creative Commons Attribution (CC BY) 4.0](https://creativecommons.org/licenses/by/4.0/). Appropriate citation for images is provided below.

## Acknowledgments

This work was in part conceived at [Funcapalooza](https://github.com/Imageomics/FuncaPalooza-2025).

This work uses data from the [Beetles as Sentinel Taxa: Predicting drought conditions from NEON specimen imagery dataset](https://huggingface.co/datasets/imageomics/sentinel-beetles). If you make use of the data, please be sure to cite the original source:

BibTeX:
```tex
@misc{East-beetles-2025,
  author = {Alyson East and Michael Belitz and Leah Cotton and Jacqueline Dominguez and Isabelle Betancourt and 
    S M Rayeed and Fangxun Liu and David Carlyn and Connor Kilrain and Jiaman Wu and Chandra Earl and Hilmar Lapp 
    and Kayla I. Perry and Charles Stewart and Matthew J. Thompson and Elizabeth G. Campolongo and Wei-Lun Chao and
    Eric R. Sokol and Sydne Record},
  title = {Beetles as Sentinel Taxa: Predicting drought conditions from {NEON} specimen imagery},
  year = {2025},
  url = {https://huggingface.co/datasets/imageomics/sentinel-beetles},
  doi = {<doi once generated>},
  publisher = {Hugging Face}
}
```

Please be sure to also cite the original data sources (specimen collection and metadata) and include the NEON acknowledgements (provided below).
```tex
@misc{NEON-pinned-beetles,
  doi = {10.48443/CD21-Q875},
  url = {https://data.neonscience.org/data-products/DP1.10022.001/RELEASE-2025},
  author = {{National Ecological Observatory Network (NEON)}},
  keywords = {diversity, taxonomy, community composition, species composition, population, invertebrates, abundance, beetles, Carabidae, insects, DNA sequences, COI, DNA barcoding, ground beetles, pitfall traps, material samples, archived samples, bet, introduced species, invasive species, native species, biodiversity},
  language = {en},
  title = {Ground beetles sampled from pitfall traps (DP1.10022.001)},
  publisher = {National Ecological Observatory Network (NEON)},
  year = {2025}
}
```
```tex
@misc{NEON-field-metadata,
  url = {https://www.neonscience.org/field-sites/exports/NEON_Field_Site_Metadata_20250625},
  author = {{National Ecological Observatory Network (NEON)}},
  language = {en},
  title = {{NEON} Field Site Metadata},
  publisher = {National Ecological Observatory Network (NEON)},
  year = {2025},
  note = {Dataset accessed from https://data.neonscience.org/api/v0/locations/sites on June 25, 2025}
}
```
