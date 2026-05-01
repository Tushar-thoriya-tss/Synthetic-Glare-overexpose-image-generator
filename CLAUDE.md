# Project: Glare / Overexposure Detection in OVD

## Context
This project is part of an **Optical Verification Document (OVD)** pipeline. Document images are captured in **uncontrolled real-world conditions**, frequently via mobile devices, so capture quality varies widely across lighting conditions, document types, and devices.

## Problem
Glare and overexposed regions — caused by reflections from light sources or camera flash — appear as very bright, washed-out patches with little to no texture or readable information. They:
- Degrade document image quality
- Hurt downstream tasks: **OCR, face matching, fraud detection**

## Goal
Reliably **identify and localize** glare / overexposed regions within document images so they can be:
- Filtered out, **or**
- Corrected, **or**
- Otherwise handled appropriately during further processing

## Key Challenge
Distinguish **true glare regions** from **naturally bright areas** of the document (e.g., white backgrounds, light-colored fields), while staying robust across:
- Different lighting conditions
- Different document types
- Different capture devices

## Working Notes for Claude
- Treat this as a **localization** problem (region/mask output), not just a binary classification.
- Robustness across devices/lighting matters more than peak accuracy on any single dataset.
- Solutions should consider that documents naturally contain bright regions — pure thresholding on luminance alone is insufficient.
