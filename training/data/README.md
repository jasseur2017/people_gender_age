## Data preparation

### CrowdHuman Dataset
- CrowdHuman Dataset is a collection of images capturing a multitude of people with their annotations Each annotation contains information about the associated person. The information is basically bounding boxes of the head, the visible body and whole body. This dataset was made specifically for people detection. We added gender and age annotations to a part of those people to extend its usability to people detection with gender and age classification.
- To download the dataset, run "crowdhuman_annotation.ipynb" notebook.

### khaliji Dataset
- Khaliji dataset is a small expansion that was added to highlight gender classification for people wearing local dress (white for male and black for female). For age classification we limited ourselves to CrowdHuman dataset.

### MOT datasets
- We used MOT datasets only for people re-identification training task neither people detection neither people age/gender classification.
- To download the datasets, run "mot_annotation.ipynb" notebook.