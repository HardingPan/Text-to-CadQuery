# Data Annotation

This folder contains scripts for annotating CAD sequences from the Text2CAD dataset using Gemini 2.0 Flash. Specifically, we convert minimal JSON-based CAD sequences into corresponding CadQuery code.

We annotated **all 170,000 CAD sequences** from Text2CAD with corresponding CadQuery programs. The full annotated dataset is available here:

👉 [CadQuery.zip](https://huggingface.co/ricemonster/NeurIPS11092/blob/main/CadQuery.zip)

## Annotation Pipeline

The following figure shows our annotation workflow:

![Annotation Pipeline](./annotation_pipeline.png)