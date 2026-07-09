<div align="center">
  
#  Football Analysis & Player Tracking

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

**A Machine Learning and Computer Vision pipeline designed to analyze football match footage, track player movements, and extract tactical insights.**

</div>

---

##  Overview
This project applies Computer Vision techniques to standard football broadcast footage. It is built to automatically detect players, track their positions across frames, and displays stats such as Pace, Distance, Possession, etc.

<div align="center">
  
![Demo of the model in action](assets/demo.gif)

</div>

## Tech Stack
* **Language:** Python, Node.JS
* **Computer Vision:** OpenCV, YOLOv5
* **Data Processing:** NumPy, Pandas, BeautifulSoup, Cheerio, Request-Promise

## 📊 Hardware & Performance Benchmarks
To give an idea of inference speed, this model was developed and tested on the following local hardware setup:
* **CPU:** Intel Core i5-11400
* **GPU:** NVIDIA GeForce RTX 3050 (4GB Laptop GPU)
* **RAM:** 16GB

## Machine Learning + Player Search/ Comparison
Users can search players and get the most 5 similar players in terms of stats.
* Stats are webscrapped from FBRef using BeautifulSoup/Cheerio and RequestPromise
* Similarity algorithm is made by using Cosine Similarity, Elbow Method, and KMeans Clustering

