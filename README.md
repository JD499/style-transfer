# Style Transfer on Spherical Images Using CNNs

### Team Members
- James Dixon  
- Robert Ace Gonzales  
- Steve Ma  
- Arye Oskotsky  

### Course Information
- **Course**: CSC 671 Deep Learning  
- **Instructor**: Dr. Robert Mateescu  
- **Institution**: San Francisco State University  
- **Date**: October 10, 2024  

---

## **Project Overview**
This project aims to apply **Style Transfer** to **spherical (360-degree) images** using **Convolutional Neural Networks (CNNs)**. Spherical images are typically represented as **equirectangular projections**, which pose challenges for traditional style transfer methods due to:
- Distortion caused by flattening spherical images.
- Difficulty in handling transitional edges.

### **Objective**
- Develop a method to perform style transfer while minimizing distortions and artifacts, particularly around the seams of the image.
- Leverage **rectilinear projections** to maintain seamless transitions across the entire spherical surface.

### **Image Dataset**
- **Source**: Microsoft COCO dataset

---

## **Project Phases**
### **Phase 1: Base CNN Style Transfer**
- Implement a CNN for style transfer on ordinary (non-spherical) images.
- Based on the work of **Gatys, Ecker, and Bethge (2016)**:  
  [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

### **Phase 2: Style Transfer on Spherical Images**
- Extend the CNN from Phase 1 to work on **spherical images**.
- Approach inspired by **Ruder, Dosovitskiy, and Brox (2018)**:  
  [Artistic Style Transfer for Videos and Spherical Images](https://arxiv.org/pdf/1708.04538)

### **Phase 3: Network Optimization**
- Perform **hyperparameter search** to optimize the network for spherical style transfer.
- Focus on minimizing artifacts, particularly near the seams of rectilinear projections.

---

## **References**
1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks*.  
   [DOI: 10.1109/CVPR.2016.265](https://doi.org/10.1109/CVPR.2016.265)

2. Ruder, M., Dosovitskiy, A., & Brox, T. (2018). *Artistic Style Transfer for Videos and Spherical Images*.  
   [DOI: 10.1007/s11263-018-1089-z](https://doi.org/10.1007/s11263-018-1089-z)

---

## **How to Run the Code**
1. Clone the repository:
   ```bash
   git clone https://github.com/JD499/style-transfer.git
   cd style-transfer